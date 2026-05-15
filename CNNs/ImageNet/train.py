# train.py
from __future__ import annotations

import gc
import math
import os
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from torch.amp import GradScaler
import torch._dynamo

from data import build_loaders
from models import build_model as build_any_model

from utils.args import parse_args
from utils import (
    ddp_setup,
    is_main_process,
    barrier,
    cleanup,
    seed_everything,
    validate_global_model_dataset_args,
    get_num_classes,
    get_in_chans,
    build_optim_sched,
    load_checkpoint,
    save_last_checkpoint,
    save_best_checkpoint,
    save_best_acc_checkpoint,
    find_latest_checkpoint,
    find_candidate_checkpoints,
)

from utils.torch_perf import configure_torch_perf
from utils.train_eval import count_params, train_one_epoch, validate
from utils.ckpt_eval import single_process_eval_checkpoints
from utils.experiment import resolve_arch_and_kwargs, get_run_dir, become_single_process_for_eval

import sys
import tqdm


def make_test_log_file(args) -> Path:
    """
    logs/<model>_<size>_<run_name>_<dataset>_ggm/test_accs.txt
    """
    logs_root = Path("logs")
    subdir = f"{args.model}_{args.size}_{args.run_name}_{args.dataset}_ggm"
    out_dir = logs_root / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "test_accs.txt"


def append_test_log(log_file: Path, line: str):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def maybe_autoresume(
    *,
    args,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    run_dir: Path,
    device: torch.device,
    is_distributed: bool,
):
    """
    Auto-resume ONLY supported when --num_runs==1, consistent with original behavior.
    Returns (start_epoch, best_loss, best_acc).
    """
    start_epoch = 0
    best_loss = float("inf")
    best_acc = -float("inf")

    if not args.resume or args.num_runs != 1:
        return start_epoch, best_loss, best_acc

    resume_path = find_latest_checkpoint(run_dir)
    if resume_path is None:
        if is_main_process(is_distributed):
            print(f"[RESUME] No checkpoint found in {run_dir}. Starting from scratch.", flush=True)
        return start_epoch, best_loss, best_acc

    if is_main_process(is_distributed):
        print(f"Auto-resuming from checkpoint: {resume_path}", flush=True)

    base_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    start_epoch, ckpt = load_checkpoint(
        base_model,
        str(resume_path),
        device,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
    )

    if scheduler is not None and isinstance(ckpt.get("scheduler", None), dict):
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            if is_main_process(is_distributed):
                print("[WARN] Could not load scheduler state_dict; continuing with fresh scheduler.", flush=True)

    best_loss = float(ckpt.get("best_loss", float("inf")))
    best_acc = float(ckpt.get("best_acc", -float("inf")))

    barrier(device)
    return start_epoch, best_loss, best_acc


def main():
    args = parse_args()
    validate_global_model_dataset_args(args)

    env = ddp_setup()
    device = env.device

    configure_torch_perf()

    torch._dynamo.config.optimize_ddp = False
    torch.set_float32_matmul_precision(args.matmul_precision)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if args.num_runs < 1:
        raise ValueError("--num_runs must be >= 1")

    if args.resume and args.num_runs > 1 and is_main_process(env.is_distributed):
        print("[WARN] --resume is only supported when --num_runs == 1. Ignoring --resume.", flush=True)

    test_log_file: Optional[Path] = None
    if is_main_process(env.is_distributed) and args.num_runs > 1:
        test_log_file = make_test_log_file(args)
        append_test_log(
            test_log_file,
            f"=== {args.dataset} | {args.model} {args.size} | run_name={args.run_name} ===",
        )

    test_accs: list[float] = []

    try:
        for run_idx in range(args.num_runs):
            seed_everything(args.seed + env.rank + run_idx * args.seed_step)

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

            arch, model_kwargs = resolve_arch_and_kwargs(
                args, num_classes=num_classes, in_chans=in_chans, device=device
            )

            model = build_any_model(arch, **model_kwargs).to(device=device)

            if device.type == "cuda":
                for p in model.parameters():
                    if p.ndim == 4:
                        p.data = p.data.to(memory_format=torch.channels_last)

            if not getattr(args, "no_compile", False) and hasattr(torch, "compile"):
                compile_kwargs = {
                    "fullgraph": bool(args.compile_fullgraph),
                    "dynamic": False,
                }

                if args.compile_mode != "none":
                    compile_kwargs["mode"] = args.compile_mode

                if is_main_process(env.is_distributed):
                    print(f"[COMPILE] torch.compile enabled with {compile_kwargs}", flush=True)

                model = torch.compile(model, **compile_kwargs)

            if is_main_process(env.is_distributed):
                print(model)
                total, trainable, frozen = count_params(model)
                print(f"[PARAMS] total={total:,} | trainable={trainable:,} | frozen={frozen:,}", flush=True)

            if env.is_distributed:
                model = nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[env.local_rank] if device.type == "cuda" else None,
                    output_device=env.local_rank if device.type == "cuda" else None,
                    broadcast_buffers=False,
                    find_unused_parameters=False,
                    gradient_as_bucket_view=True,
                )

            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

            opt_sched = build_optim_sched(args.dataset, model, trainloader, args, env.world_size)
            optimizer = opt_sched.optimizer
            scheduler = opt_sched.scheduler
            step_sched_per_update = opt_sched.step_scheduler_per_update

            use_amp = bool(args.amp) and (device.type == "cuda")
            if args.amp and device.type != "cuda" and is_main_process(env.is_distributed):
                print("[WARN] --amp set but device is not CUDA; AMP disabled.", flush=True)
            scaler = GradScaler("cuda") if use_amp else None

            run_dir = get_run_dir(args, run_idx)
            run_dir.mkdir(parents=True, exist_ok=True)

            if args.test:
                become_single_process_for_eval(env, device)

                print(f"\n[TEST-ONLY] Evaluating checkpoints in: {run_dir}", flush=True)
                ckpt_paths = find_candidate_checkpoints(run_dir)

                if not ckpt_paths:
                    last_path = run_dir / "last.pth"
                    if last_path.exists():
                        ckpt_paths = [last_path]
                        print(f"[TEST-ONLY][FALLBACK] No best checkpoints found; using {last_path}", flush=True)
                    else:
                        latest = find_latest_checkpoint(run_dir)
                        if latest is not None:
                            ckpt_paths = [latest]
                            print(f"[TEST-ONLY][FALLBACK] No best checkpoints found; using latest: {latest}", flush=True)
                        else:
                            print(f"[TEST-ONLY][WARN] No checkpoints found in {run_dir}", flush=True)
                            cleanup()
                            return

                best_path, best_loss, best_acc = single_process_eval_checkpoints(
                    args=args,
                    arch=arch,
                    model_kwargs=model_kwargs,
                    device=device,
                    ckpt_paths=ckpt_paths,
                )
                print(f"[TEST-ONLY] BEST checkpoint by TEST acc: {best_path} | acc: {best_acc:.2f}%", flush=True)
                cleanup()
                return

            start_epoch, best_loss, best_acc = maybe_autoresume(
                args=args,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                run_dir=run_dir,
                device=device,
                is_distributed=env.is_distributed,
            )
            best_loss_path: Optional[Path] = None
            best_acc_path: Optional[Path] = None

            if is_main_process(env.is_distributed) and args.num_runs > 1:
                print(f"\n========== RUN {run_idx+1}/{args.num_runs} ==========", flush=True)

            for epoch in range(start_epoch, args.epochs):
                if env.is_distributed and train_sampler is not None:
                    train_sampler.set_epoch(epoch)

                if is_main_process(env.is_distributed):
                    print(f"\nEpoch {epoch+1}/{args.epochs} (world_size={env.world_size})", flush=True)

                epoch_start = time.time()

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
                )

                val_loss, val_acc = validate(
                    model=model,
                    loader=valloader,
                    criterion=criterion,
                    device=device,
                    is_distributed=env.is_distributed,
                )

                if scheduler is not None and not step_sched_per_update:
                    scheduler.step()

                epoch_time = time.time() - epoch_start
                current_lr = optimizer.param_groups[0]["lr"]

                if is_main_process(env.is_distributed):
                    print(
                        f"[{epoch_time:.1f}s] LR: {current_lr:.6f} | "
                        f"Train loss: {train_loss:.4f}, acc: {train_acc:.2f}% | "
                        f"Val loss: {val_loss:.4f}, acc: {val_acc:.2f}%",
                        flush=True,
                    )

                    halfway_epoch = args.epochs // 2
                    allow_best_saving = (epoch + 1) > halfway_epoch

                    if allow_best_saving:
                        if val_loss < best_loss:
                            old = best_loss
                            best_loss = val_loss
                            best_loss_path = save_best_checkpoint(
                                run_dir=run_dir,
                                epoch=epoch,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                arch=arch,
                                dataset=args.dataset,
                                best_loss=best_loss,
                                prev_best_path=best_loss_path,
                                scaler=scaler,
                            )
                            print(
                                f"[BEST_LOSS] val_loss {old:.4f} -> {best_loss:.4f}  saving {best_loss_path}",
                                flush=True,
                            )

                        if val_acc > best_acc:
                            old = best_acc
                            best_acc = val_acc
                            best_acc_path = save_best_acc_checkpoint(
                                run_dir=run_dir,
                                epoch=epoch,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                arch=arch,
                                dataset=args.dataset,
                                best_acc=best_acc,
                                prev_best_path=best_acc_path,
                                scaler=scaler,
                            )
                            print(
                                f"[BEST_ACC] val_acc {old:.2f}% -> {best_acc:.2f}%  saving {best_acc_path}",
                                flush=True,
                            )

                    save_last_checkpoint(
                        run_dir=run_dir,
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        arch=arch,
                        dataset=args.dataset,
                        best_loss=best_loss,
                        best_acc=best_acc,
                        scaler=scaler,
                    )

                barrier(device)

            become_single_process_for_eval(env, device)

            if args.dataset == "imagenet":
                candidates = []
                if best_acc_path is not None:
                    candidates.append(best_acc_path)
            else:
                candidates = []
                if best_loss_path is not None:
                    candidates.append(best_loss_path)
                if best_acc_path is not None:
                    candidates.append(best_acc_path)

            if not candidates:
                print("\n[WARN] No best checkpoints found to evaluate.", flush=True)
            else:
                best_path, best_test_loss, best_test_acc = single_process_eval_checkpoints(
                    args=args,
                    arch=arch,
                    model_kwargs=model_kwargs,
                    device=device,
                    ckpt_paths=candidates,
                )

                test_accs.append(float(best_test_acc))
                if test_log_file is not None:
                    append_test_log(
                        test_log_file,
                        f"run {run_idx+1:02d}: {best_test_acc:.2f}%  (chosen_ckpt={best_path.name})",
                    )

            del model, optimizer, scheduler, trainloader, valloader, testloader, train_sampler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        if is_main_process(env.is_distributed):
            if not test_accs:
                print("\nNo test results collected.", flush=True)
            else:
                mean = sum(test_accs) / len(test_accs)
                std = 0.0
                if len(test_accs) > 1:
                    var = sum((x - mean) ** 2 for x in test_accs) / (len(test_accs) - 1)
                    std = math.sqrt(var)

                print("\n========== SUMMARY ==========", flush=True)
                for i, acc in enumerate(test_accs, start=1):
                    print(f"run {i:02d}: {acc:.2f}%", flush=True)
                print(f"mean ± std: {mean:.2f}% ± {std:.2f}%  (n={len(test_accs)})", flush=True)

                if test_log_file is not None:
                    append_test_log(test_log_file, "---")
                    for i, acc in enumerate(test_accs, start=1):
                        append_test_log(test_log_file, f"run {i:02d}: {acc:.2f}%")
                    append_test_log(test_log_file, f"mean ± std: {mean:.2f}% ± {std:.2f}%  (n={len(test_accs)})")
                    append_test_log(test_log_file, "")

    finally:
        cleanup()


if __name__ == "__main__":
    main()