from __future__ import annotations

import glob
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def _atomic_torch_save(obj: Any, path: str) -> None:
    """
    Atomic-ish save: write to a temp file in the same directory, then os.replace().
    Uses a non-dot temp filename + .pth suffix to avoid PyTorchFileWriter 'invalid file name'
    on some filesystems.
    """
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix="tmp_ckpt_", suffix=".pth", dir=out_dir)
    os.close(fd)

    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    _atomic_torch_save(state, path)


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    strict: bool = True,
) -> Tuple[int, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model"], strict=strict)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    return start_epoch, ckpt


def _get_state_dict(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def save_last_checkpoint(
    run_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    arch: str,
    dataset: str,
    best_loss: float,
    best_acc: float,
) -> None:
    state = {
        "epoch": epoch,
        "arch": arch,
        "dataset": dataset,
        "model": _get_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss,
        "best_acc": best_acc,
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()

    save_checkpoint(state, str(run_dir / "last.pth"))


def save_best_checkpoint(
    run_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    arch: str,
    dataset: str,
    best_loss: float,
    prev_best_path: Optional[Path],
) -> Path:
    if prev_best_path is not None and prev_best_path.exists():
        os.remove(prev_best_path)

    state = {
        "epoch": epoch,
        "arch": arch,
        "dataset": dataset,
        "model": _get_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss,
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()

    new_best_path = run_dir / f"best_loss_e{epoch+1:03d}.pth"
    save_checkpoint(state, str(new_best_path))
    return new_best_path


def save_best_acc_checkpoint(
    run_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    arch: str,
    dataset: str,
    best_acc: float,
    prev_best_path: Optional[Path],
) -> Path:
    if prev_best_path is not None and prev_best_path.exists():
        os.remove(prev_best_path)

    state = {
        "epoch": epoch,
        "arch": arch,
        "dataset": dataset,
        "model": _get_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "best_acc": best_acc,
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()

    new_best_path = run_dir / f"best_acc_e{epoch+1:03d}.pth"
    save_checkpoint(state, str(new_best_path))
    return new_best_path


def find_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    last = run_dir / "last.pth"
    if last.exists():
        return last
    
    checkpoints = sorted(
        glob.glob(str(run_dir / "*.pth")),
        key=os.path.getmtime,
        reverse=True
    )
    return Path(checkpoints[0]) if checkpoints else None

def find_candidate_checkpoints(run_dir: Path) -> list[Path]:
    """Finds best_loss_* and best_acc_* checkpoints."""
    candidates = []
    candidates.extend(run_dir.glob("best_loss_*.pth"))
    candidates.extend(run_dir.glob("best_acc_*.pth"))
    return sorted(list(set(c for c in candidates if c.exists())))