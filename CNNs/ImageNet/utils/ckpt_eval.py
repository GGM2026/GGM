# utils/ckpt_eval.py
from __future__ import annotations

import gc
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from data import build_loaders
from models import build_model as build_any_model
from utils.train_eval import validate


def strip_state_dict_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        if k.startswith("module."):
            k = k[len("module.") :]
        out[k] = v
    return out


def single_process_eval_checkpoints(
    *,
    args,
    arch: str,
    model_kwargs: dict,
    device: torch.device,
    ckpt_paths: list[Path],
) -> Tuple[Path, float, float]:
   
    _, _, testloader_single, _, _, _ = build_loaders(
        dataset=args.dataset,
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_distributed=False,  # critical
        img_size=args.img_size,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
    )

    eval_criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    def eval_one_ckpt(ckpt_path: Path) -> tuple[float, float]:
        m = build_any_model(arch, **model_kwargs)
        m.to(device)

        ckpt = torch.load(str(ckpt_path), map_location=device)
        sd = ckpt.get("model", ckpt)
        sd = strip_state_dict_prefixes(sd)

        missing, unexpected = m.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[WARN] {ckpt_path.name}: missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print("  missing (first 10):", missing[:10])
            if unexpected:
                print("  unexpected (first 10):", unexpected[:10])

        tl, ta = validate(
            model=m,
            loader=testloader_single,
            criterion=eval_criterion,
            device=device,
            is_distributed=False,
        )

        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return tl, ta

    best_acc = -1.0
    best_loss = 0.0
    best_path = ckpt_paths[0]

    print("\nEvaluating checkpoints on TEST (single-process):", flush=True)
    for p in ckpt_paths:
        tl, ta = eval_one_ckpt(p)
        print(f"[TEST] {p.name} | loss: {tl:.4f}, acc: {ta:.2f}%", flush=True)
        if ta > best_acc:
            best_acc = ta
            best_loss = tl
            best_path = p

    print(f"\n[TEST][CHOSEN] {best_path.name} | loss: {best_loss:.4f}, acc: {best_acc:.2f}%", flush=True)
    return best_path, best_loss, best_acc