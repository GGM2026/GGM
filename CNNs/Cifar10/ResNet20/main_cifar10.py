from __future__ import annotations

import os
import csv
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from resnet_build import build_resnet_cifar


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def seed_everything(seed: int = 1337, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def configure_torch_perf() -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--run_name", type=str, default="cifar10")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--results_file", type=str, default="")
    p.add_argument("-r", "--resume", action="store_true")
    p.add_argument("-t", "--test", action="store_true")

    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--deterministic", action="store_true")

    p.add_argument("-dr", "--data_root", type=str, required=True)
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--split_seed", type=int, default=1337)

    p.add_argument("-s", "--size", type=str, default="18", choices=["18", "20"])

    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("-lr", "--base_lr", type=float, default=1e-3)
    p.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    p.add_argument("-ls", "--label_smoothing", type=float, default=0.0)
    p.add_argument("-n", "--N_scale", type=float, default=1.0)

    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")

    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no-compile", dest="compile", action="store_false")

    p.add_argument("--use_ggm", action="store_true", default=True)
    p.add_argument("--no-ggm", dest="use_ggm", action="store_false")

    return p


def cifar_transforms(img_size: int = 32):
    train_tf = T.Compose([
        T.RandomCrop(img_size, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
    ])

    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_tf, eval_tf


def _split_indices(n: int, val_fraction: float, seed: int) -> Tuple[list[int], list[int]]:
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be between 0 and 1")

    g = torch.Generator()
    g.manual_seed(seed)

    perm = torch.randperm(n, generator=g).tolist()
    val_len = int(round(n * val_fraction))
    val_len = max(1, min(val_len, n - 1))

    val_idx = perm[:val_len]
    train_idx = perm[val_len:]
    return train_idx, val_idx


def build_loaders(
    root: str,
    batch_size: int,
    num_workers: int,
    img_size: int = 32,
    val_fraction: float = 0.1,
    split_seed: int = 1337,
    drop_last: bool = True,
):
    train_tf, eval_tf = cifar_transforms(img_size)

    train_base = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
    val_base = datasets.CIFAR10(root=root, train=True, download=True, transform=eval_tf)
    test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=eval_tf)

    train_idx, val_idx = _split_indices(len(train_base), val_fraction, split_seed)
    train_ds = Subset(train_base, train_idx)
    val_ds = Subset(val_base, val_idx)

    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        drop_last=drop_last,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return train_loader, val_loader, test_loader


def build_model(args, device: torch.device) -> nn.Module:
    model = build_resnet_cifar(
        model_name=args.size,
        num_classes=10,
        in_chans=3,
        use_ggm=args.use_ggm,
        N_scale=args.N_scale,
    )

    model = model.to(device=device, memory_format=torch.channels_last)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    return model


@dataclass
class OptimSched:
    optimizer: torch.optim.Optimizer
    scheduler: Optional[Any]
    step_scheduler_per_update: bool


def build_optim_sched(model: nn.Module, train_loader, args) -> OptimSched:
    max_lr = args.base_lr * (args.batch_size / 256.0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=len(train_loader) * args.epochs,
        pct_start=0.1,
    )

    return OptimSched(optimizer, scheduler, step_scheduler_per_update=True)


def count_params(model: nn.Module) -> Tuple[int, int, int]:
    total = 0
    trainable = 0
    frozen = 0

    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        else:
            frozen += n

    return total, trainable, frozen


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    scheduler: Optional[Any] = None,
    step_scheduler_per_update: bool = False,
    scaler: Optional[GradScaler] = None,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad(set_to_none=True)
    use_amp = scaler is not None

    it = tqdm(loader, total=len(loader), desc="Training", leave=False)

    for images, targets in it:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        if scheduler is not None and step_scheduler_per_update:
            scheduler.step()

        running_loss += float(loss.item()) * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        acc = 100.0 * correct / max(total, 1)
        it.set_postfix(loss=f"{running_loss / max(total, 1):.4f}", acc=f"{acc:.2f}%")

    avg_loss = running_loss / max(total, 1.0)
    epoch_acc = 100.0 * correct / max(total, 1.0)
    return avg_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    desc: str = "Eval",
):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    it = tqdm(loader, total=len(loader), desc=desc, leave=False)

    for images, targets in it:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += float(loss.item()) * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        acc = 100.0 * correct / max(total, 1)
        it.set_postfix(loss=f"{running_loss / max(total, 1):.4f}", acc=f"{acc:.2f}%")

    avg_loss = running_loss / max(total, 1.0)
    acc = 100.0 * correct / max(total, 1.0)
    return avg_loss, acc


def get_run_dir(args) -> Path:
    return Path(args.checkpoint_dir) / args.run_name


def get_checkpoint_paths(run_dir: Path) -> tuple[Path, Path]:
    last_path = run_dir / "last.pth"
    best_path = run_dir / "best.pth"
    return last_path, best_path


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_val_acc: float,
    args,
    scaler: Optional[GradScaler] = None,
) -> None:
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_val_acc": best_val_acc,
        "args": vars(args),
    }

    if scaler is not None:
        state["scaler"] = scaler.state_dict()

    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler: Optional[GradScaler] = None,
):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_val_acc = float(ckpt.get("best_val_acc", -1.0))
    return start_epoch, best_val_acc


def append_test_result(results_file: str, args, test_loss: float, test_acc: float, best_val_acc: float) -> None:
    if not results_file:
        return

    path = Path(results_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "run_name",
                "seed",
                "split_seed",
                "N_scale",
                "epochs",
                "test_loss",
                "test_acc",
                "best_val_acc",
            ])
        writer.writerow([
            args.run_name,
            args.seed,
            args.split_seed,
            args.N_scale,
            args.epochs,
            f"{test_loss:.6f}",
            f"{test_acc:.4f}",
            f"{best_val_acc:.4f}",
        ])


def main() -> None:
    args = build_parser().parse_args()

    seed_everything(args.seed, deterministic=args.deterministic)
    configure_torch_perf()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = get_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt, best_ckpt = get_checkpoint_paths(run_dir)

    train_loader, val_loader, test_loader = build_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
    )

    model = build_model(args, device)

    total, trainable, frozen = count_params(model)
    print(f"[PARAMS] total={total:,} | trainable={trainable:,} | frozen={frozen:,}", flush=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    opt_sched = build_optim_sched(model, train_loader, args)
    optimizer = opt_sched.optimizer
    scheduler = opt_sched.scheduler
    step_sched_per_update = opt_sched.step_scheduler_per_update

    use_amp = bool(args.amp) and device.type == "cuda"
    if args.amp and device.type != "cuda":
        print("[WARN] AMP requested but CUDA is unavailable. Running without AMP.", flush=True)

    scaler = GradScaler("cuda") if use_amp else None

    start_epoch = 0
    best_val_acc = -1.0

    if args.resume:
        if last_ckpt.exists():
            start_epoch, best_val_acc = load_checkpoint(
                last_ckpt,
                model,
                device,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
            print(f"[RESUME] Loaded checkpoint: {last_ckpt}", flush=True)
        else:
            print(f"[WARN] No checkpoint found at {last_ckpt}", flush=True)

    if args.test:
        ckpt_path = best_ckpt if best_ckpt.exists() else last_ckpt
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint found in {run_dir}")

        _, best_val_acc = load_checkpoint(ckpt_path, model, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, desc="Test")
        print(f"[TEST] loss={test_loss:.4f} | acc={test_acc:.2f}%", flush=True)
        append_test_result(args.results_file, args, test_loss, test_acc, best_val_acc)
        return

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}", flush=True)

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            step_scheduler_per_update=step_sched_per_update,
            scaler=scaler,
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            desc="Validation",
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"LR: {current_lr:.6f} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}% | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%",
            flush=True,
        )

        save_checkpoint(
            last_ckpt,
            epoch,
            model,
            optimizer,
            scheduler,
            best_val_acc,
            args,
            scaler=scaler,
        )

        if val_acc > best_val_acc:
            old = best_val_acc
            best_val_acc = val_acc
            save_checkpoint(
                best_ckpt,
                epoch,
                model,
                optimizer,
                scheduler,
                best_val_acc,
                args,
                scaler=scaler,
            )
            print(f"[BEST] val_acc {old:.2f}% -> {best_val_acc:.2f}%", flush=True)

    if best_ckpt.exists():
        _, best_val_acc = load_checkpoint(best_ckpt, model, device)
        print(f"\nLoaded best checkpoint: {best_ckpt}", flush=True)
    elif last_ckpt.exists():
        _, best_val_acc = load_checkpoint(last_ckpt, model, device)
        print(f"\nLoaded last checkpoint: {last_ckpt}", flush=True)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device, desc="Test")
    print(f"[TEST] loss={test_loss:.4f} | acc={test_acc:.2f}%", flush=True)
    append_test_result(args.results_file, args, test_loss, test_acc, best_val_acc)


if __name__ == "__main__":
    main()