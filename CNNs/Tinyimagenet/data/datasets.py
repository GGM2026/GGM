from __future__ import annotations
from typing import Tuple
import os
from torchvision.datasets import ImageNet, CIFAR10, CIFAR100, FashionMNIST,ImageFolder


def build_dataset(name: str, root: str, train: bool, transform):
    name = name.lower()

    if name == "imagenet":
        split = "train" if train else "val"
        return ImageNet(root=root, split=split, transform=transform)

    if name == "cifar10":
        return CIFAR10(root=root, train=train, download=True, transform=transform)

    if name == "cifar100":
        return CIFAR100(root=root, train=train, download=True, transform=transform)

    if name == "fashionmnist":
        return FashionMNIST(root=root, train=train, download=True, transform=transform)

    if name == "tinyimagenet":
        
        folder_name = "tiny-imagenet-200"
        split = "train" if train else "val"
        path = os.path.join(root, folder_name, split)
        
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"TinyImageNet path not found: {path}\n"
                f"Did you run 'python scripts/prepare_tinyimagenet.py'?"
            )
        return ImageFolder(root=path, transform=transform)    
    raise ValueError(f"Unknown dataset: {name}")


