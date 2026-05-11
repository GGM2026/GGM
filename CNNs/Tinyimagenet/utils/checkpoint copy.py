from __future__ import annotations

import os
import tempfile
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
