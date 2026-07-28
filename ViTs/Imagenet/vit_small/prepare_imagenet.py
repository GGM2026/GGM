#!/usr/bin/env python
"""
Download ImageNet-1k and organize it into the ImageFolder layout the code expects:

    imagenet/train/<class_idx>/*.JPEG
    imagenet/val/<class_idx>/*.JPEG

Class folders are named by the canonical ImageNet-1k label index (000..999), so
torchvision.ImageFolder assigns exactly those indices -- matching the checkpoint's
1000-way classification head.

Default source is the official `ILSVRC/imagenet-1k` on the Hugging Face Hub, which
is gated: run `hf auth login` once and accept the terms at
https://huggingface.co/datasets/ILSVRC/imagenet-1k . To use a non-gated mirror
instead, pass `--dataset <name>`.

Examples
--------
    python prepare_imagenet.py                       # train + val -> ./imagenet
    python prepare_imagenet.py --splits validation   # val only (for inference)
    python prepare_imagenet.py --limit-per-split 100 # tiny subset (smoke test)
"""
import argparse
import os
import sys

from datasets import load_dataset
from tqdm import tqdm

# HF split name -> local directory name the DataHandler / evaluate.py expect
SPLIT_DIRNAME = {"train": "train", "validation": "val", "val": "val", "test": "test"}


def prepare_split(dataset, hf_split, out_root, limit=None,
                  image_key="image", label_key="label"):
    out_dir = os.path.join(out_root, SPLIT_DIRNAME.get(hf_split, hf_split))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{hf_split}] streaming '{dataset}' -> {out_dir}", flush=True)

    # streaming=True writes images as they arrive (no giant local arrow cache)
    ds = load_dataset(dataset, split=hf_split, streaming=True)

    counts = {}
    for i, ex in enumerate(tqdm(ds, desc=hf_split, unit="img")):
        if limit and i >= limit:
            break
        label = int(ex[label_key])
        cls_dir = os.path.join(out_dir, f"{label:03d}")
        os.makedirs(cls_dir, exist_ok=True)
        n = counts.get(label, 0)
        counts[label] = n + 1
        ex[image_key].convert("RGB").save(
            os.path.join(cls_dir, f"{label:03d}_{n:06d}.JPEG"), quality=95)

    print(f"[{hf_split}] wrote {sum(counts.values())} images "
          f"across {len(counts)} classes", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="./imagenet", help="output root (default ./imagenet)")
    ap.add_argument("--dataset", default="ILSVRC/imagenet-1k", help="HF dataset id")
    ap.add_argument("--splits", default="train,validation",
                    help="comma-separated HF splits to fetch (default train,validation)")
    ap.add_argument("--limit-per-split", type=int, default=0,
                    help="stop after N images per split (0 = all; for testing)")
    args = ap.parse_args()

    try:
        for sp in [s.strip() for s in args.splits.split(",") if s.strip()]:
            prepare_split(args.dataset, sp, args.output, args.limit_per_split or None)
    except Exception as e:
        msg = str(e)
        if any(k in msg.lower() for k in ("gated", "401", "403", "authenticate", "login")):
            sys.exit(
                "\nERROR: this dataset is gated. Run `hf auth login` and accept "
                f"the terms at https://huggingface.co/datasets/{args.dataset}\n"
                "(or pass --dataset <non-gated-mirror>).\n\nDetails: " + msg[:300])
        raise

    print("\nDone. Layout ready at:", os.path.abspath(args.output))

    # HuggingFace streaming spawns background prefetch threads whose teardown can
    # trigger a GIL error ("PyGILState_Release ... Aborted") during interpreter
    # shutdown. All images are already written to disk at this point, so exit
    # immediately and cleanly instead of running the crashy finalization.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
