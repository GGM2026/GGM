# utils/checkpoint.py
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

def _strip_state_dict_prefixes(sd: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in sd.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("module."):
            k = k[len("module."):]
        out[k] = v
    return out


def _unwrap_model_for_state_dict(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def _atomic_torch_save(obj: Any, path: str) -> None:
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

def save_last_checkpoint(
    run_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    arch: str,
    dataset: str,
    best_loss: float,
    best_acc: float, 
    scaler: Optional[Any] = None,
    filename: str = "last.pth",
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / filename

    base_model = _unwrap_model_for_state_dict(model)
    state: Dict[str, Any] = {
        "epoch": epoch,
        "arch": arch,
        "dataset": dataset,
        "best_loss": best_loss,
        "best_acc": best_acc, 
        "model": base_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": (scheduler.state_dict() if scheduler is not None else None),
        "scaler": (scaler.state_dict() if scaler is not None else None),
    }

    save_checkpoint(state, str(last_path)) 
    return last_path

def save_best_checkpoint(
    run_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    arch: str,
    dataset: str,
    best_loss: float,
    prev_best_path: Optional[Path],
    scaler: Optional[Any] = None,
) -> Path:
    
    run_dir.mkdir(parents=True, exist_ok=True)

    best_path = run_dir / f"best_epoch_{epoch+1:03d}.pth"

    base_model = _unwrap_model_for_state_dict(model)
    state: Dict[str, Any] = {
        "epoch": epoch,
        "arch": arch,
        "dataset": dataset,
        "best_loss": best_loss,
        "model": base_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": (scheduler.state_dict() if scheduler is not None else None),
        "scaler": (scaler.state_dict() if scaler is not None else None),
    }

    save_checkpoint(state, str(best_path))

    if prev_best_path is not None and prev_best_path.exists() and prev_best_path != best_path:
        try:
            prev_best_path.unlink()
        except OSError:
            pass

    return best_path

def save_best_acc_checkpoint(
    run_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    arch: str,
    dataset: str,
    best_acc: float,
    prev_best_path: Optional[Path],
    scaler: Optional[Any] = None,
    filename_prefix: str = "best_acc_epoch",
) -> Path:

    run_dir.mkdir(parents=True, exist_ok=True)

    best_path = run_dir / f"{filename_prefix}_{epoch+1:03d}.pth"

    base_model = _unwrap_model_for_state_dict(model)
    state: Dict[str, Any] = {
        "epoch": epoch,
        "arch": arch,
        "dataset": dataset,
        "best_acc": best_acc,
        "model": base_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": (scheduler.state_dict() if scheduler is not None else None),
        "scaler": (scaler.state_dict() if scaler is not None else None),
    }

    save_checkpoint(state, str(best_path))

    if prev_best_path is not None and prev_best_path.exists() and prev_best_path != best_path:
        try:
            prev_best_path.unlink()
        except OSError:
            pass

    return best_path


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    strict: bool = True,
) -> Tuple[int, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)

    base_model = _unwrap_model_for_state_dict(model)

    sd = ckpt.get("model", ckpt) 
    sd = _strip_state_dict_prefixes(sd)

    missing, unexpected = base_model.load_state_dict(sd, strict=strict)

    if (len(missing) or len(unexpected)) and strict:
        raise RuntimeError(
            f"State dict mismatch after prefix stripping. "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and isinstance(ckpt.get("scheduler", None), dict):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and isinstance(ckpt.get("scaler", None), dict):
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    return start_epoch, ckpt


def find_latest_checkpoint(run_dir: Path, device: Optional[torch.device] = None) -> Optional[Path]:

    if not run_dir.exists():
        return None

    pths = sorted(run_dir.glob("*.pth"))
    if not pths:
        return None

    def is_loadable(p: Path) -> bool:
        if device is None:
            return p.exists()
        try:
            torch.load(str(p), map_location=device, weights_only=False)
            return True
        except Exception:
            return False

    last_ckpts = sorted(
        [p for p in pths if p.name.startswith("last")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in last_ckpts:
        if is_loadable(p):
            return p

    epoch_re = re.compile(r"(best(?:_acc)?_epoch)_(\d+)\.pth$")
    ranked = []
    for p in pths:
        m = epoch_re.search(p.name)
        if m and is_loadable(p):
            ranked.append((int(m.group(2)), p))

    if ranked:
        ranked.sort(key=lambda t: t[0], reverse=True)
        return ranked[0][1]

    for p in sorted(pths, key=lambda p: p.stat().st_mtime, reverse=True):
        if is_loadable(p):
            return p

    return None

from typing import Iterable, List

@torch.no_grad()
def evaluate_checkpoint(
    model: nn.Module,
    loader,
    device: torch.device,
    ckpt_path: str | Path,
    criterion: Optional[nn.Module] = None,
    strict: bool = True,
) -> Dict[str, float]:

    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    base_model = _unwrap_model_for_state_dict(model)
    sd = ckpt.get("model", ckpt)
    sd = _strip_state_dict_prefixes(sd)
    base_model.load_state_dict(sd, strict=strict)
    base_model.to(device)
    base_model.eval()

    total = 0
    correct = 0
    running_loss = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = base_model(images)

        if criterion is not None:
            loss = criterion(outputs, targets)
            running_loss += float(loss.item()) * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    acc = 100.0 * correct / max(total, 1)

    out = {"acc": float(acc)}
    if criterion is not None:
        out["loss"] = float(running_loss / max(total, 1))
    return out


@torch.no_grad()
def evaluate_checkpoints_choose_best(
    model: nn.Module,
    loader,
    device: torch.device,
    ckpt_paths: Iterable[str | Path],
    criterion: Optional[nn.Module] = None,
    strict: bool = True,
    verbose: bool = True,
) -> Tuple[Path, Dict[str, float], List[Tuple[Path, Dict[str, float]]]]:
   
    results: List[Tuple[Path, Dict[str, float]]] = []

    best_path: Optional[Path] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_acc = -1.0

    for p in ckpt_paths:
        p = Path(p)
        metrics = evaluate_checkpoint(
            model=model,
            loader=loader,
            device=device,
            ckpt_path=p,
            criterion=criterion,
            strict=strict,
        )
        results.append((p, metrics))

        if verbose:
            if "loss" in metrics:
                print(f"[EVAL] {p.name} | loss: {metrics['loss']:.4f}, acc: {metrics['acc']:.2f}%")
            else:
                print(f"[EVAL] {p.name} | acc: {metrics['acc']:.2f}%")

        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            best_path = p
            best_metrics = metrics

    assert best_path is not None and best_metrics is not None, "No checkpoints were evaluated."
    return best_path, best_metrics, results


def find_candidate_checkpoints(run_dir: Path) -> list[Path]:
    
    run_dir = Path(run_dir)

    def latest_by_epoch(glob_pat: str, regex_pat: str) -> Optional[Path]:
        paths = list(run_dir.glob(glob_pat))
        if not paths:
            return None
        rx = re.compile(regex_pat)
        best: tuple[int, Path] | None = None
        for p in paths:
            m = rx.search(p.name)
            if not m:
                continue
            ep = int(m.group(1))
            if best is None or ep > best[0]:
                best = (ep, p)
        return None if best is None else best[1]

    latest_best_loss = latest_by_epoch(
        glob_pat="best_epoch_*.pth",
        regex_pat=r"best_epoch_(\d+)\.pth$",
    )
    latest_best_acc = latest_by_epoch(
        glob_pat="best_acc_epoch_*.pth",
        regex_pat=r"best_acc_epoch_(\d+)\.pth$",
    )

    cands: list[Path] = []
    if latest_best_loss is not None:
        cands.append(latest_best_loss)
    if latest_best_acc is not None and (latest_best_loss is None or latest_best_acc.resolve() != latest_best_loss.resolve()):
        cands.append(latest_best_acc)

    last = run_dir / "last.pth"
    if last.exists():
        if not any(p.resolve() == last.resolve() for p in cands):
            cands.append(last)

    return cands

