# data/loaders.py
from __future__ import annotations
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from .datasets import build_dataset
from .transforms import imagenet_transforms, cifar_transforms, fashionmnist_transforms, tinyimagenet_transforms


def build_transforms(dataset: str, img_size: int):
    d = dataset.lower()
    if d == "imagenet":
        return imagenet_transforms(img_size)
    if d == "tinyimagenet": 
        return tinyimagenet_transforms(img_size if img_size else 64)
    if d in ("cifar10", "cifar100"):
        return cifar_transforms(img_size)
    if d == "fashionmnist":
        return fashionmnist_transforms(img_size)
    raise ValueError(f"Unknown dataset: {dataset}")


def _split_indices(n: int, val_fraction: float, seed: int) -> Tuple[list[int], list[int]]:
    val_fraction = float(val_fraction)
    assert 0.0 < val_fraction < 1.0

    g = torch.Generator()
    g.manual_seed(int(seed))

    perm = torch.randperm(n, generator=g).tolist()
    val_len = int(round(n * val_fraction))
    val_len = max(1, min(val_len, n - 1))  # keep both splits non-empty

    val_idx = perm[:val_len]
    train_idx = perm[val_len:]
    return train_idx, val_idx


def build_loaders(
    dataset: str,
    root: str,
    batch_size: int,
    num_workers: int,
    is_distributed: bool,
    img_size: int = 224,
    drop_last: bool = True,
    val_fraction: float = 0.1,   # NEW
    split_seed: int = 1337,      # NEW (must be same on all ranks)
):
    """
    Returns:
      train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler
    """
    train_tf, val_tf = build_transforms(dataset, img_size)
    d = dataset.lower()

    # --- Build datasets in a way that val uses val_tf but shares the same underlying examples ---
    if d == "imagenet":
        # Standard ImageNet protocol:
        # - train on official ImageNet train split
        # - validate/checkpoint/report on official ImageNet val split
        # - no public labeled test set, so keep test_ds as official val only for API compatibility
        train_ds = build_dataset(d, root, train=True, transform=train_tf)
        val_ds   = build_dataset(d, root, train=False, transform=val_tf)
        test_ds  = build_dataset(d, root, train=False, transform=val_tf)
    
    elif d == "tinyimagenet":
        # Keep your previous TinyImageNet behavior.
        # Split TRAIN into train+val; use official VAL as TEST.
        train_base = build_dataset(d, root, train=True, transform=train_tf)
        val_base   = build_dataset(d, root, train=True, transform=val_tf)
        test_ds    = build_dataset(d, root, train=False, transform=val_tf)
    
        train_idx, val_idx = _split_indices(
            len(train_base),
            val_fraction=val_fraction,
            seed=split_seed,
        )
        train_ds = Subset(train_base, train_idx)
        val_ds   = Subset(val_base, val_idx)

    else:
        # CIFAR/FashionMNIST: split TRAIN into train+val; use official TEST as test.
        train_base = build_dataset(d, root, train=True, transform=train_tf)
        val_base   = build_dataset(d, root, train=True, transform=val_tf)
        test_ds    = build_dataset(d, root, train=False, transform=val_tf)

        train_idx, val_idx = _split_indices(len(train_base), val_fraction=val_fraction, seed=split_seed)
        train_ds = Subset(train_base, train_idx)
        val_ds   = Subset(val_base, val_idx)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    val_sampler   = DistributedSampler(val_ds, shuffle=False) if is_distributed else None
    test_sampler  = None

    # If num_workers==0, persistent_workers must be False
    persistent = (num_workers > 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        drop_last=drop_last,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler
