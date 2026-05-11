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
    """
    Reads Adam-style betas/eps if present in args; otherwise uses safe defaults.
    Keeps this module robust even if you don't add the optional CLI flags.
    """
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
    """
    Paper-clean optimizer support: only AdamW and SGD.

    Expected args:
      --optimizer {adamw,sgd}
      --weight_decay float

    SGD extras:
      --momentum float
      --nesterov

    AdamW extras (optional):
      --betas b1 b2
      --eps eps
    """
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
    """
    Builds Optimizer and Scheduler.
    Controlled by args.scheduler (options: 'onecycle', 'cosine').
    
    Defaults if args.scheduler is not set:
      - CIFAR/FashionMNIST -> OneCycleLR
      - ImageNet -> CosineAnnealingLR
    """
    d = dataset.lower()
    global_batch = args.batch_size * world_size

    
    sched_type = getattr(args, "scheduler", None)
    
    if not sched_type: 
        if d == "imagenet":
            sched_type = "cosine"
        else:
            sched_type = "onecycle"
            
    sched_type = sched_type.lower().strip()

   
    if sched_type == "onecycle":
        if d in ("fashionmnist", "cifar10", "cifar100"):
            lr = args.base_lr * (global_batch / 256.0)
        else:
            lr = args.base_lr * world_size

        opt = build_optimizer(args, model, lr=lr)

        updates_per_epoch = _num_optimizer_updates_per_epoch(
            len(train_loader), args.accumulation_steps
        )

        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=lr,
            total_steps=updates_per_epoch * args.epochs,
            pct_start=0.1,
        )
        return OptimSched(opt, sched, step_scheduler_per_update=True)

   
    elif sched_type == "cosine":
       
        if d == "imagenet":
            lr = args.base_lr * world_size
        else:
            lr = args.base_lr 

        opt = build_optimizer(args, model, lr=lr)

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=args.epochs,
            eta_min=1e-6,
        )
        return OptimSched(opt, sched, step_scheduler_per_update=False)

    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}. Supported: ['onecycle', 'cosine']")