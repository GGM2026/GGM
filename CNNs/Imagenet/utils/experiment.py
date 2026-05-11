# utils/experiment.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.distributed as dist

from utils.model_config import resolve_timm_backbone
from utils.ddp import barrier, cleanup


def resolve_arch_and_kwargs(
    args,
    *,
    num_classes: int,
    in_chans: int,
    device: torch.device,
) -> Tuple[str, dict]:
    model_family = args.model.lower()

    if model_family == "resnet":
        size = args.size or "18"
        if size not in {"18", "20", "34"}:
            raise ValueError(f"--size for resnet must be 18/20/34, got {args.size}")

        timm_backbone = resolve_timm_backbone("resnet", size)

        if args.double_residual and size not in {"18", "34"}:
            # warning printed in main only on rank0; keep logic there if you want
            pass

        if args.GGM:
            if size == "20":
                raise ValueError(
                    "--GGM is not supported with --size 20. ResNet-20 is a custom CIFAR "
                    "architecture not available in timm. Use --size 18 or --size 34 with --GGM."
                )
            arch = "GGM_resnet"
            model_kwargs = dict(
                model_name=timm_backbone,
                num_classes=num_classes,
                pretrained=False,
                requires_grad=True,
                device=device,
                img_size=args.img_size,
                in_chans=in_chans,
                dataset_name=args.dataset,
                use_prelu=args.prelu,
                prelu_init=0.25,
                replace_blocks_with_convbn=not args.full_precision,
                use_double_residual=args.double_residual,
                full_precision=args.full_precision,
            )
            return arch, model_kwargs

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
            use_prelu=args.prelu,
        )
        return arch, model_kwargs

    if model_family == "vgg":
        size = (args.size or "16").lower().strip()
        if size not in {"small", "11", "16"}:
            raise ValueError(f"--size for vgg must be small/11/16, got {args.size}")

        model_name = "vgg16"
        if size == "small":
            model_name = "vgg_small"
        elif size == "11":
            model_name = "vgg11"

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
        return arch, model_kwargs

    raise ValueError(f"Unknown --model '{args.model}'")


def get_run_dir(args, run_idx: int) -> Path:
    if args.num_runs == 1:
        return Path(args.ckpt_dir) / args.run_name
    return Path(args.ckpt_dir) / args.run_name / f"run{run_idx+1:02d}"


def become_single_process_for_eval(env, device: torch.device) -> None:
    """
    Ensures only rank0 continues, and destroys the process group so no collectives can occur.
    """
    barrier(device)

    if env.is_distributed and env.rank != 0:
        cleanup()
        raise SystemExit(0)

    if env.is_distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()