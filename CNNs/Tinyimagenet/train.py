from __future__ import annotations

import argparse
import math
import os
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast 
import torch.distributed as dist
from contextlib import nullcontext
from tqdm import tqdm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from utils import (
    ddp_setup,
    is_main_process,
    all_reduce_sum,
    load_checkpoint,
    barrier,
    cleanup,
    seed_everything,
    save_last_checkpoint,
    save_best_checkpoint,
    save_best_acc_checkpoint,
    find_latest_checkpoint,
    find_candidate_checkpoints,
    Logger,
)

from utils.dataset_meta import get_num_classes, get_in_chans
from utils.model_config import (
    resolve_timm_backbone,
    validate_global_model_dataset_args,
    validate_vgg_args,
)
from utils.optim import build_optim_sched

from data import build_loaders
from models import build_model as build_any_model


def count_params(model: nn.Module):
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
    mixup_fn: Optional[Mixup] = None,
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

    for i, (images, targets) in it:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        do_step = ((i + 1) % accumulation_steps == 0) or ((i + 1) == len(loader))
        
        if ddp_wrapped and not do_step:
            cm = model.no_sync()
        else:
            cm = nullcontext()

        with cm:
            with autocast(device_type="cuda", enabled=(scaler is not None)): 
                outputs = model(images)
                loss = criterion(outputs, targets) / max(1, accumulation_steps)
            
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        running_loss += float(loss.item()) * max(1, accumulation_steps) * images.size(0)

        if mixup_fn is not None:
            _, preds = outputs.max(1)
            _, target_indices = targets.max(1)
            correct += preds.eq(target_indices).sum().item()
        else:
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()

        total += targets.size(0)

        if do_step:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None and step_scheduler_per_update:
                scheduler.step()

        if is_main_process(is_distributed):
            acc = 100.0 * correct / max(total, 1)
            it.set_postfix(loss=f"{(running_loss/max(total,1)):.4f}", acc=f"{acc:.2f}%")

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
        images = images.to(device, non_blocking=True)
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


def resample_all_G(model, epoch, every=10, is_distributed=False, rank=0):
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


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, required=True, choices=["imagenet", "cifar10", "cifar100", "fashionmnist","tinyimagenet"])
    p.add_argument("--run_name", type=str, default="exp")
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--pretrained", type=str, default="", help="Path to pre-trained weights (init only, no resume).")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--split_seed", type=int, default=1337)
    p.add_argument("--model", type=str, required=True, choices=["resnet", "vgg", "simple"])
    p.add_argument("--size", type=str, default="")
    p.add_argument("--N_scale", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--batch_size", type=int, default=16) 
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--base_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--full_precision", action="store_true", help="Use standard full-precision timm layers.")
    p.add_argument("--chunk_N", type=int, default=0, help="Chunk size for GGD Conv (0 to disable)")
    p.add_argument("--num_runs", type=int, default=1)
    p.add_argument("--seed_step", type=int, default=1000)
    p.add_argument("--optimizer", type=str, default="adamw")
    p.add_argument("--scheduler", type=str, default=None)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", action="store_true")
    p.add_argument("--amp", action="store_true")

    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--cutmix", type=float, default=0.0)
    p.add_argument("--mixup_prob", type=float, default=1.0)
    p.add_argument("--drop_path", type=float, default=0.0)

    return p.parse_args()


def main():
    args = parse_args()

    if args.dataset.lower() == "tinyimagenet":
        if ddp_setup(peek=True).is_main:
             print(f"[INFO] Tiny ImageNet detected. Overriding --img_size to 64.")
        args.img_size = 64

    validate_global_model_dataset_args(args)

    env = ddp_setup()
    device = env.device

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    if args.num_runs < 1:
        raise ValueError("--num_runs must be >= 1")

    if args.resume and args.num_runs > 1 and is_main_process(env.is_distributed):
        print("[WARN] --resume is only supported when --num_runs == 1. Ignoring --resume.", flush=True)

    test_accs = [] 

    try:
        for run_idx in range(args.num_runs):
            seed_everything(args.seed + env.rank + run_idx * args.seed_step)

            if args.num_runs == 1:
                run_dir = Path(args.ckpt_dir) / args.run_name
            else:
                run_dir = Path(args.ckpt_dir) / args.run_name / f"run{run_idx+1:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            logger = None
            if is_main_process(env.is_distributed):
                logger = Logger(str(run_dir))

            trainloader, valloader, testloader, train_sampler, _, _ = build_loaders(
                dataset=args.dataset,
                root=args.data_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                is_distributed=env.is_distributed,
                img_size=args.img_size,
                val_fraction=args.val_fraction,
                split_seed=args.split_seed,
            )

            num_classes = get_num_classes(args.dataset)
            in_chans = get_in_chans(args.dataset)

            mixup_fn = None
            mixup_active = args.mixup > 0 or args.cutmix > 0.
            if mixup_active:
                mixup_fn = Mixup(
                    mixup_alpha=args.mixup, 
                    cutmix_alpha=args.cutmix, 
                    cutmix_minmax=None,
                    prob=args.mixup_prob, 
                    switch_prob=0.5, 
                    mode='batch',
                    label_smoothing=args.label_smoothing, 
                    num_classes=num_classes
                )

            model_family = args.model.lower()
            if model_family == "resnet":
                if args.size == "": args.size = "18"
                timm_backbone = resolve_timm_backbone("resnet", args.size)
                
                arch = "ggd_resnet"
                model_kwargs = dict(
                    model_name=timm_backbone,
                    num_classes=num_classes,
                    N_scale=args.N_scale,
                    requires_grad=True,
                    device=device,
                    img_size=args.img_size,
                    in_chans=in_chans,
                    dataset_name=args.dataset,
                    full_precision=args.full_precision,
                    chunk_N=args.chunk_N,
                    drop_path_rate=args.drop_path,
                )

            elif model_family == "vgg":
                if args.size == "": args.size = "medium"
                vgg_size = (args.size or "medium").lower().strip()
                model_name = f"vgg_{vgg_size}"
                arch = "ggd_vgg"
                model_kwargs = dict(
                    model_name=model_name,
                    num_classes=num_classes,
                    N_scale=args.N_scale,
                    requires_grad=True,
                    device=device,
                    img_size=args.img_size,
                    in_chans=in_chans,
                    dataset_name=args.dataset,
                    full_precision=args.full_precision,
                    chunk_N=args.chunk_N,
                )
            elif model_family == "simple":
                arch = "simple"
                model_kwargs = dict(
                    net_type=args.size,
                    num_classes=num_classes,
                    N_scale=args.N_scale
                )
            else:
                raise ValueError(f"Unknown --model '{args.model}'")

            
            model = build_any_model(arch, **model_kwargs)
            model = model.to(device)
            if args.pretrained and os.path.isfile(args.pretrained):
                if is_main_process(env.is_distributed):
                    print(f"[INIT] Loading GGM weights from: {args.pretrained}", flush=True)
                
                ckpt = torch.load(args.pretrained, map_location=device)
                state_dict = ckpt['model'] if 'model' in ckpt else ckpt
                
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                
                missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
                
                if is_main_process(env.is_distributed):
                    print(f"[INIT] Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}", flush=True)

            if is_main_process(env.is_distributed):
                total, trainable, frozen = count_params(model)
                print(f"[PARAMS] total={total:,} | trainable={trainable:,} | frozen={frozen:,}", flush=True)

            if env.is_distributed:
                model = nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[env.local_rank] if device.type == "cuda" else None,
                    output_device=env.local_rank if device.type == "cuda" else None,
                    broadcast_buffers=False,
                    find_unused_parameters=False,
                )

            if mixup_active:
                criterion = SoftTargetCrossEntropy()
            else:
                criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

            opt_sched = build_optim_sched(args.dataset, model, trainloader, args, env.world_size)
            optimizer = opt_sched.optimizer
            scheduler = opt_sched.scheduler
            step_sched_per_update = opt_sched.step_scheduler_per_update
            scaler = GradScaler() if args.amp else None
            
            if is_main_process(env.is_distributed):
                print(f"Automatic Mixed Precision (AMP) {'enabled' if scaler else 'disabled'}.")
            
            start_epoch = 0
            best_loss = float("inf")
            best_path: Optional[Path] = None

            if args.resume and args.num_runs == 1 and os.path.isfile(args.resume):
                if is_main_process(env.is_distributed):
                    print(f"Resuming from checkpoint: {args.resume}", flush=True)

                base_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
                start_epoch, ckpt = load_checkpoint(
                    base_model,
                    args.resume,
                    device,
                    optimizer=optimizer,
                    scheduler=None,
                    scaler=scaler,
                )
                if scheduler is not None and isinstance(ckpt.get("scheduler", None), dict):
                    try: scheduler.load_state_dict(ckpt["scheduler"])
                    except: pass
                best_loss = float(ckpt.get("best_loss", float("inf")))
                best_path = Path(args.resume)
                barrier(device) 

            if is_main_process(env.is_distributed):
                if args.num_runs > 1:
                    print(f"\n========== RUN {run_idx+1}/{args.num_runs} ==========", flush=True)

            for epoch in range(start_epoch, args.epochs):
                if env.is_distributed and train_sampler is not None:
                    train_sampler.set_epoch(epoch)
              
                if args.dataset.lower() != "tinyimagenet":
                    if (args.epochs > 150) and (epoch < 150): resample_all_G(model, epoch, every=5, is_distributed=env.is_distributed, rank=env.rank)
                    if (args.epochs < 100) and (epoch < 60): resample_all_G(model, epoch, every=5, is_distributed=env.is_distributed, rank=env.rank)

                train_loss, train_acc = train_one_epoch(
                    model=model,
                    loader=trainloader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    accumulation_steps=args.accumulation_steps,
                    is_distributed=env.is_distributed,
                    scheduler=scheduler,
                    step_scheduler_per_update=step_sched_per_update,
                    scaler=scaler,
                    mixup_fn=mixup_fn,
                )

                val_loss, val_acc = validate(model, valloader, nn.CrossEntropyLoss(), device, env.is_distributed)

                if scheduler is not None and not step_sched_per_update:
                    scheduler.step()

                current_lr = optimizer.param_groups[0]["lr"]

                if is_main_process(env.is_distributed):
                    print(
                        f"LR: {current_lr:.6f} | "
                        f"Train loss: {train_loss:.4f}, acc: {train_acc:.2f}% | "
                        f"Val loss: {val_loss:.4f}, acc: {val_acc:.2f}%",
                        flush=True,
                    )
                    if logger is not None:
                        logger.log(epoch + 1, train_loss, train_acc, val_loss, val_acc, current_lr)
                    if val_loss < best_loss:
                        old = best_loss
                        best_loss = val_loss
                        best_path = save_best_checkpoint(
                            run_dir=run_dir,
                            epoch=epoch,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            arch=arch,
                            dataset=args.dataset,
                            best_loss=best_loss,
                            prev_best_path=best_path,
                            scaler=scaler,
                        )
                        print(f"[BEST] val_loss {old:.4f} -> {best_loss:.4f}  saving {best_path}", flush=True)

                barrier(device)

            barrier(device)

            if is_main_process(env.is_distributed) and best_path is not None:
                print(f"\nEvaluating best checkpoint on TEST: {best_path}", flush=True)
                base_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
                ckpt = torch.load(str(best_path), map_location=device)
                base_model.load_state_dict(ckpt["model"], strict=True)

                test_loss, test_acc = validate(base_model, testloader, nn.CrossEntropyLoss(), device, False)
                print(f"[TEST][run {run_idx+1}] loss: {test_loss:.4f}, acc: {test_acc:.2f}%", flush=True)
                test_accs.append(float(test_acc))
                
                if logger is not None:
                    logger.save_plots(final_test_acc=float(test_acc))
                    logger.save_mat()

            barrier(device)

            del model, optimizer, scheduler, trainloader, valloader, testloader, train_sampler
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        if is_main_process(env.is_distributed):
            if len(test_accs) == 0:
                print("\nNo test results collected.", flush=True)
            else:
                mean = sum(test_accs) / len(test_accs)
                print(f"\n========== SUMMARY ==========")
                for i, acc in enumerate(test_accs, start=1):
                    print(f"run {i:02d}: {acc:.2f}%")
                print(f"mean: {mean:.2f}%")

        barrier(device)

    finally:
        cleanup()


if __name__ == "__main__":
    main()