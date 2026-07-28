"""
Evaluate the provided 1-bit GGM ViT checkpoint on ImageNet-1k validation.

Reproduces the reference result of the compressed 4.27 MiB checkpoint:
    Top-1 = 56.19%,  Top-5 = 79.61%   (ImageNet-1k val, 50k images)

Two modes:
  * Full ImageNet   :  --data_dir /path/to/imagenet/val   (ImageFolder layout)
  * Quick sanity    :  --imagenette                        (10-class subset, no
                       ImageNet download; auto-downloads Imagenette if absent)

The checkpoint-loading logic (FrozenLinearGGM + load_compressed_model) is copied
verbatim from imagenet_checkpoint_eval.ipynb.
"""

import argparse
import os
import tarfile
import urllib.request
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from main import Config
from model import VisionTransformer
from linearggm import LinearGGM


# Imagenette WNID -> canonical ImageNet-1k class index (sorted-WNID ordering,
# the ordering torchvision.ImageFolder and HF imagenet-1k both use).
IMAGENETTE_TO_IMAGENET = {
    "n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491,
    "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571,
    "n03445777": 574, "n03888257": 701,
}
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"


# ---------------------------------------------------------------------------
# Verbatim from imagenet_checkpoint_eval.ipynb
# ---------------------------------------------------------------------------
class FrozenLinearGGM(nn.Module):
    def __init__(self, *, in_features, out_features, N, centralize_x, G_seed,
                 G, W_b, gain, bias=None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.N = int(N)
        self.centralize_x = bool(centralize_x)
        self.G_seed = int(G_seed)
        self.register_buffer("W_b", W_b.to(torch.int8).contiguous())
        self.register_buffer("gain", gain.to(torch.float32).contiguous())
        self.register_buffer("G", G.to(torch.float32).contiguous(), persistent=False)
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.to(torch.float32).contiguous())

    def forward(self, x):
        x_eff = x - x.mean(dim=-1, keepdim=True) if self.centralize_x else x.float()
        x_b = (x_eff @ self.G.transpose(-1, -2)).sign()
        y = (x_b @ self.W_b.to(x_b.dtype)) / self.N
        y = y * self.gain
        return y if self.bias is None else y + self.bias


def load_checkpoint_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def replace_submodule(root, module_name, replacement):
    if "." in module_name:
        parent_name, child_name = module_name.rsplit(".", 1)
        parent = root.get_submodule(parent_name)
    else:
        parent, child_name = root, module_name
    setattr(parent, child_name, replacement)


def unpack_binary_tensor(packed, metadata):
    packed = packed.detach().cpu().to(torch.uint8).flatten()
    shifts = torch.arange(8, dtype=torch.int16)
    bits = (((packed.to(torch.int16)[:, None] >> shifts[None, :]) & 1).flatten())
    bits = bits[: metadata["numel"]]
    return (bits.to(torch.int8) * 2 - 1).reshape(metadata["shape"]).contiguous()


def build_frozen_architecture(artifact):
    model = VisionTransformer(Config(**artifact["config"]))
    use_exact_g = artifact["g_storage"] == "exact_deduplicated"
    for module_name, spec in artifact["linear_ggm_specs"].items():
        original_layer = model.get_submodule(module_name)
        if not isinstance(original_layer, LinearGGM):
            raise TypeError(f"{module_name} is not a LinearGGM in the base model")
        if use_exact_g:
            G = artifact["g_bank"][artifact["g_map"][module_name]].cpu().contiguous()
        else:
            if int(original_layer.G_seed) != int(spec["G_seed"]):
                raise RuntimeError(f"G seed mismatch for {module_name}")
            G = original_layer.G.detach().cpu().contiguous()
        replacement = FrozenLinearGGM(
            in_features=spec["in_features"], out_features=spec["out_features"],
            N=spec["N"], centralize_x=spec["centralize_x"], G_seed=spec["G_seed"],
            G=G, W_b=torch.zeros(spec["N"], spec["out_features"], dtype=torch.int8),
            gain=torch.ones(spec["out_features"], dtype=torch.float32),
            bias=(torch.zeros(spec["out_features"], dtype=torch.float32)
                  if spec["has_bias"] else None),
        )
        replace_submodule(model, module_name, replacement)
    return model


def load_compressed_model(path, target_device):
    artifact = load_checkpoint_cpu(path)
    model = build_frozen_architecture(artifact)
    if artifact["wb_storage"] != "packed_1bit":
        raise ValueError(f"Expected packed_1bit checkpoint, got {artifact['wb_storage']}")
    result = model.load_state_dict(artifact["state_dict"], strict=False)
    expected_missing = {f"{m}.W_b" for m in artifact["linear_ggm_specs"]}
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint loading failed. Missing: {result.missing_keys}; "
            f"unexpected: {result.unexpected_keys}")
    with torch.no_grad():
        for module_name in artifact["linear_ggm_specs"]:
            layer = model.get_submodule(module_name)
            layer.W_b.copy_(unpack_binary_tensor(
                artifact["packed_wb"][module_name],
                artifact["packed_wb_metadata"][module_name]))
    return model.to(target_device).eval(), Config(**artifact["config"])
# ---------------------------------------------------------------------------


def build_val_transform(image_size):
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def maybe_download_imagenette(root):
    val_dir = os.path.join(root, "imagenette2-320", "val")
    if os.path.isdir(val_dir):
        return val_dir
    os.makedirs(root, exist_ok=True)
    tgz = os.path.join(root, "imagenette2-320.tgz")
    print(f"Downloading Imagenette to {tgz} ...")
    urllib.request.urlretrieve(IMAGENETTE_URL, tgz)
    print("Extracting ...")
    with tarfile.open(tgz) as t:
        t.extractall(root)
    os.remove(tgz)
    return val_dir


@torch.inference_mode()
def evaluate(model, loader, device, use_amp, remap=None, limit_batches=0):
    top1 = top5 = total = 0
    amp = (torch.autocast(device_type="cuda", dtype=torch.float16)
           if use_amp and device.type == "cuda" else nullcontext())
    for b, (images, labels) in enumerate(tqdm(loader, desc="Evaluating")):
        if limit_batches and b >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        if remap is not None:
            labels = remap[labels]
        labels = labels.to(device, non_blocking=True)
        with amp:
            logits = model(images)
        t5 = logits.topk(5, dim=1).indices
        top1 += (t5[:, 0] == labels).sum().item()
        top5 += t5.eq(labels[:, None]).any(dim=1).sum().item()
        total += labels.size(0)
    return 100.0 * top1 / total, 100.0 * top5 / total, total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="runs/best_seed_G_frozen_Wb_1bit_fp16.pth")
    p.add_argument("--data_dir", default="./imagenet/val",
                   help="ImageNet val directory (ImageFolder: <data_dir>/<class>/*.JPEG); "
                        "populate it with `python prepare_imagenet.py --splits validation`")
    p.add_argument("--imagenette", action="store_true",
                   help="Quick sanity check on Imagenette (auto-downloads; no ImageNet needed)")
    p.add_argument("--imagenette_root", default="./data")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit_batches", type=int, default=0,
                   help="Evaluate only the first N batches (0 = all)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint} "
          f"({os.path.getsize(args.checkpoint) / 1024**2:.2f} MiB)")
    model, config = load_compressed_model(args.checkpoint, device)
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params).")

    tf = build_val_transform(config.image_size)
    remap = None

    if args.imagenette:
        val_dir = maybe_download_imagenette(args.imagenette_root)
        dataset = datasets.ImageFolder(val_dir, transform=tf)
        remap = torch.tensor([IMAGENETTE_TO_IMAGENET[w]
                              for w in sorted(dataset.class_to_idx)])
        print(f"Imagenette quick check: {len(dataset)} images (10 classes).")
    else:
        if not args.data_dir or not os.path.isdir(args.data_dir):
            raise SystemExit(
                "Provide --data_dir <imagenet_val> (ImageFolder layout) "
                "or use --imagenette for a no-download sanity check.")
        dataset = datasets.ImageFolder(args.data_dir, transform=tf)
        # prepare_imagenet.py names folders by ImageNet class index (000..999).
        # ImageFolder assigns labels by *sorted position*, which only equals the
        # true class index when all 1000 folders are present. If the folder names
        # are numeric, read the class index straight from the name so evaluation
        # is correct even on a partial subset of classes.
        if all(name.isdigit() for name in dataset.classes):
            remap = torch.empty(len(dataset.classes), dtype=torch.long)
            for name, pos in dataset.class_to_idx.items():
                remap[pos] = int(name)
            print(f"ImageNet val: {len(dataset)} images across "
                  f"{len(dataset.classes)} class folders "
                  f"(labels read from folder index).")
        else:
            print(f"ImageNet val: {len(dataset)} images ({len(dataset.classes)} classes).")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")

    top1, top5, n = evaluate(model, loader, device, use_amp, remap, args.limit_batches)
    print(f"\nEvaluated {n} images")
    print(f"Top-1: {top1:.2f}%")
    print(f"Top-5: {top5:.2f}%")
    if not args.imagenette and not args.limit_batches:
        print("Reference (full ImageNet val): Top-1 56.19%, Top-5 79.61%")


if __name__ == "__main__":
    main()
