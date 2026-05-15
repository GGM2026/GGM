# utils/optim.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class OptimSched:
    optimizer: torch.optim.Optimizer
    scheduler: Optional[Any]
    step_scheduler_per_update: bool  


def _num_optimizer_updates_per_epoch(num_batches: int, accumulation_steps: int) -> int:
    return int(math.ceil(num_batches / max(1, accumulation_steps)))


def _get_betas_eps(args) -> Tuple[Tuple[float, float], float]:
    betas = (0.9, 0.999)
    eps = 1e-8

    if hasattr(args, "betas") and args.betas is not None:
        b = args.betas
        if isinstance(b, (list, tuple)) and len(b) == 2:
            betas = (float(b[0]), float(b[1]))

    if hasattr(args, "eps") and args.eps is not None:
        eps = float(args.eps)

    return betas, eps


def build_optimizer(args, model: nn.Module, lr: float) -> torch.optim.Optimizer:
    name = str(getattr(args, "optimizer", "adamw")).lower().strip()
    wd = float(getattr(args, "weight_decay", 0.0))

    if name == "adamw":
        betas, eps = _get_betas_eps(args)
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            betas=betas,
            eps=eps,
        )

    if name == "sgd":
        momentum = float(getattr(args, "momentum", 0.9))
        nesterov = bool(getattr(args, "nesterov", False))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            momentum=momentum,
            nesterov=nesterov,
        )

    raise ValueError(f"Unknown optimizer: {name}. Allowed: ['adamw', 'sgd']")


def build_optim_sched(
    dataset: str,
    model: nn.Module,
    train_loader,
    args,
    world_size: int,
) -> OptimSched:
    d = dataset.lower()

    if d in ("fashionmnist", "cifar10", "cifar100"):
        global_batch = args.batch_size * world_size
        max_lr = args.base_lr * (global_batch / 256.0)

        opt = build_optimizer(args, model, lr=max_lr)

        updates_per_epoch = _num_optimizer_updates_per_epoch(
            len(train_loader), args.accumulation_steps
        )

        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=max_lr,
            total_steps=updates_per_epoch * args.epochs,
            pct_start=0.1,
        )
        return OptimSched(opt, sched, step_scheduler_per_update=True)

    if d in ("imagenet", "tinyimagenet"):
        global_batch = args.batch_size * world_size
        lr = args.base_lr * (global_batch / 256.0)
    
        opt = build_optimizer(args, model, lr=lr)
    
        updates_per_epoch = _num_optimizer_updates_per_epoch(
            len(train_loader), args.accumulation_steps
        )
        total_updates = updates_per_epoch * args.epochs
    
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=lr,
            total_steps=total_updates,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=1000.0,
        )
    
        return OptimSched(opt, sched, step_scheduler_per_update=True)

    raise ValueError(f"No optimizer/scheduler setup defined for dataset={dataset}")