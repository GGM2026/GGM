# data/datasets.py
from __future__ import annotations
from typing import Tuple
import os  

from torchvision.datasets import ImageNet, CIFAR10, CIFAR100, FashionMNIST
from torchvision.datasets import ImageFolder 

def build_dataset(name: str, root: str, train: bool, transform):
    name = name.lower()

    if name == "imagenet":
        split = "train" if train else "val"
        return ImageNet(root=root, split=split, transform=transform)

    if name == "tinyimagenet":  
        base = os.path.join(root, "tiny-imagenet-200")
        # train=True -> train/, train=False -> val_fixed/ (official val used as TEST)
        split_dir = "train" if train else "val_fixed"
        return ImageFolder(root=os.path.join(base, split_dir), transform=transform)

    if name == "cifar10":
        return CIFAR10(root=root, train=train, download=True, transform=transform)

    if name == "cifar100":
        return CIFAR100(root=root, train=train, download=True, transform=transform)

    if name == "fashionmnist":
        return FashionMNIST(root=root, train=train, download=True, transform=transform)

    raise ValueError(f"Unknown dataset: {name}")

