from __future__ import annotations
from typing import Tuple

import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

FASHION_MEAN = (0.2860,)
FASHION_STD  = (0.3530,)


def imagenet_transforms(img_size: int = 224):
    train = T.Compose([
        T.RandomResizedCrop(img_size),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.2),
    ])
    val = T.Compose([
        T.Resize(int(img_size * 256 / 224)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, val

def tinyimagenet_transforms(img_size: int = 64):
    train = T.Compose([
        
        T.RandomCrop(img_size, padding=8, padding_mode='reflect'),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.25),
    ])

    val = T.Compose([
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, val


def cifar_transforms(img_size: int = 32):
    train = T.Compose([
        T.RandomCrop(img_size, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
    ])
    val = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train, val



def fashionmnist_transforms(img_size: int=28):
    train = T.Compose([
        T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(FASHION_MEAN, FASHION_STD),
        T.RandomErasing(p=0.1, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
    ])
    val = T.Compose([
        T.ToTensor(),
        T.Normalize(FASHION_MEAN, FASHION_STD),
    ])
    return train, val