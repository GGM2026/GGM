# utils/train_eval.py
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from utils.ddp import is_main_process
from utils.ddp import all_reduce_sum


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
    accumulation_steps: int,
    is_distributed: bool,
    scheduler: Optional[Any] = None,
    step_scheduler_per_update: bool = False,
    scaler: Optional[GradScaler] = None,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad(set_to_none=True)

    it = enumerate(loader)
    if is_main_process(is_distributed):
        it = tqdm(it, total=len(loader), desc="Training", leave=False)

    ddp_wrapped = isinstance(model, nn.parallel.DistributedDataParallel)
    use_amp = scaler is not None

    for i, (images, targets) in it:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        do_step = ((i + 1) % max(1, accumulation_steps) == 0) or ((i + 1) == len(loader))

        sync_ctx = nullcontext()
        if ddp_wrapped and not do_step:
            sync_ctx = model.no_sync()

        with sync_ctx:
            with autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets) / max(1, accumulation_steps)

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        running_loss += float(loss.item()) * max(1, accumulation_steps) * images.size(0)

        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        if do_step:
            if use_amp:
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                new_scale = scaler.get_scale()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None and step_scheduler_per_update:
                scheduler.step()

        if is_main_process(is_distributed):
            acc = 100.0 * correct / max(total, 1)
            it.set_postfix(loss=f"{(running_loss / max(total, 1)):.4f}", acc=f"{acc:.2f}%")

    if is_distributed:
        running_loss = all_reduce_sum(running_loss, device)
        correct = all_reduce_sum(correct, device)
        total = all_reduce_sum(total, device)

    avg_loss = running_loss / max(total, 1.0)
    epoch_acc = 100.0 * correct / max(total, 1.0)
    return avg_loss, epoch_acc


@torch.no_grad()
def validate(model: nn.Module, loader, criterion, device: torch.device, is_distributed: bool):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    it = loader
    if is_main_process(is_distributed):
        it = tqdm(loader, desc="Validation", leave=False)

    for images, targets in it:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += float(loss.item()) * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    if is_distributed:
        running_loss = all_reduce_sum(running_loss, device)
        correct = all_reduce_sum(correct, device)
        total = all_reduce_sum(total, device)

    avg_loss = running_loss / max(total, 1.0)
    acc = 100.0 * correct / max(total, 1.0)
    return avg_loss, acc


def resample_all_G(model, epoch: int, every: int = 10, is_distributed: bool = False, rank: int = 0) -> None:
    if every <= 0 or (epoch % every) != 0:
        return

    base = model.module if hasattr(model, "module") else model

    if (not is_distributed) or rank == 0:
        for m in base.modules():
            if hasattr(m, "resample_G"):
                m.resample_G()

    if is_distributed:
        for m in base.modules():
            if hasattr(m, "G") and isinstance(getattr(m, "G"), torch.Tensor):
                dist.broadcast(m.G, src=0)