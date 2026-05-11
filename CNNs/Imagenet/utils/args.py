# utils/args.py
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import yaml

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # dataset
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["imagenet", "tinyimagenet", "cifar10", "cifar100", "fashionmnist"],
    )

    # experiment
    p.add_argument("--run_name", type=str, default="exp")
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    p.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="If set, auto-resume from the most recent .pth in {ckpt_dir}/{run_name}.",
    )

    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--img_size", type=int, default=224)

    p.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of the training split reserved for validation.",
    )
    p.add_argument("--split_seed", type=int, default=1337)

    # model selection
    p.add_argument("--model", type=str, required=True, choices=["resnet", "vgg"])
    p.add_argument(
        "--size",
        type=str,
        default="",
        help="Model size. ResNet: 18/20/34. VGG: small/11/16.",
    )

    p.add_argument("--N_scale", type=float, default=1.0)

    # training
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--batch_size", type=int, default=None,
                    help="Per-GPU batch size. Auto-loaded from system_config.yaml if not set (fallback: 16).")
    p.add_argument("--num_workers", type=int, default=None,
                    help="DataLoader workers. Auto-loaded from system_config.yaml if not set (fallback: 8).")
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--base_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.1)

    p.add_argument(
        "-fp",
        "--full_precision",
        action="store_true",
        help="Disable GGD conv swapping (use plain nn.Conv2d).",
    )
    p.add_argument("--prelu", action="store_true", help="Use channel-wise PReLU (where supported).")

    p.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of independent training runs (different seeds) to estimate mean±std of test accuracy.",
    )
    p.add_argument(
        "--seed_step",
        type=int,
        default=1000,
        help="Seed offset between runs: run_seed = seed + rank + run_idx*seed_step",
    )

    # optimizer
    p.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "sgd"],
        help="Optimizer to use. Scheduler remains dataset-specific.",
    )
    p.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    p.add_argument("--nesterov", action="store_true", help="Use Nesterov momentum for SGD")

    # modes
    p.add_argument(
        "--test",
        action="store_true",
        help="Test-only: skip training and evaluate existing checkpoints under ckpt_dir/run_name.",
    )

    # AMP / variants
    p.add_argument("--amp", action="store_true", default=True,
                    help="Enable AMP training (default: enabled).")
    p.add_argument("--no-amp", dest="amp", action="store_false",
                    help="Disable AMP training.")
    p.add_argument("--no-compile", dest="no_compile", action="store_true",
                    help="Disable torch.compile (useful for debugging or incompatible ops).")
    p.add_argument(
        "--GGM",
        action="store_true",
        help="If set, use GGM ConvBN replacements inside ResNet blocks (stem stays nn.Conv2d).",
    )
    p.add_argument(
        "--double_residual",
        action="store_true",
        help="Use the 'double residual add' BasicBlock forward (ResNet18/34 only).",
    )

    return p


def parse_args(argv: list[str] | None = None):
    parser = build_parser()

    # First pass: parse to find which args the user explicitly provided.
    args = parser.parse_args(argv)

    # Detect which tunable keys were explicitly passed on the CLI by
    # comparing parsed values against their argparse defaults.
    explicit_keys: set[str] = set()
    for dest in ("batch_size", "num_workers", "chunk_N", "N_scale"):
        default_value = parser.get_default(dest)
        if getattr(args, dest, None) != default_value:
            explicit_keys.add(dest)

    # Apply system_config.yaml defaults for anything not explicitly set.
    # _apply_system_config(args, explicit_keys)

    # # Final fallback defaults for anything still None (no CLI, no config file).
    # for _yaml_key, (dest, fallback) in _TUNABLE_DEFAULTS.items():
    #     if getattr(args, dest, None) is None:
    #         setattr(args, dest, fallback)

    return args