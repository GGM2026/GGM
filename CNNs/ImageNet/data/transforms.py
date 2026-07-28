# data/transforms.py
from __future__ import annotations
from typing import Tuple

import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

FASHION_MEAN = (0.2860,)
FASHION_STD  = (0.3530,)

def imagenet_transforms(img_size: int = 224, autoaug: bool = True, re_prob: float = 0.1):
    eval_size = 224
    resize_size = 256

    train_ops = [
        T.RandomResizedCrop(eval_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.RandomHorizontalFlip(),
    ]

    if autoaug:
        train_ops += [T.RandAugment(num_ops=2, magnitude=9)]
    else:
        train_ops += [T.ColorJitter(0.15, 0.15, 0.15, 0.1)]

    train_ops += [
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=re_prob, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    ]
    train = T.Compose(train_ops)

    val = T.Compose([
        T.Resize(resize_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.CenterCrop(eval_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train, val
    
# def imagenet_transforms(img_size: int = 224, autoaug: bool = True, re_prob: float = 0.1):
#     train_ops = [
#         T.RandomResizedCrop(img_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
#         T.RandomHorizontalFlip(),
#     ]

#     if autoaug:
#         train_ops += [T.RandAugment(num_ops=2, magnitude=9)]

#     train_ops += [
#         T.ToTensor(),
#         T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
#     ]

#     if re_prob > 0:
#         train_ops += [T.RandomErasing(p=re_prob, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random")]

#     train = T.Compose(train_ops)

#     val = T.Compose([
#         T.Resize(int(img_size * 256 / 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
#         T.CenterCrop(img_size),
#         T.ToTensor(),
#         T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
#     ])

#     return train, val

# def imagenet_transforms(img_size: int = 224):
#     """
#     TorchVision reference-style ImageNet transforms for ResNet baseline.
#     - Train: RandomResizedCrop + RandomHorizontalFlip
#     - Val: Resize(256/224 * img_size) + CenterCrop
#     """
#     train = T.Compose([
#         T.RandomResizedCrop(img_size),   # default interpolation is fine for baseline
#         T.RandomHorizontalFlip(),
#         T.ToTensor(),
#         T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
#     ])

#     val_resize = int(round(img_size * 256 / 224))  # 224->256, 192->219, etc.
#     val = T.Compose([
#         T.Resize(val_resize),
#         T.CenterCrop(img_size),
#         T.ToTensor(),
#         T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
#     ])

#     return train, val

def tinyimagenet_transforms(img_size: int = 64):
    train = T.Compose([
        T.RandomResizedCrop(img_size),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.2),
    ])
    val = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, val


# def cifar_transforms(img_size: int = 32):
#     train = T.Compose([
#         T.RandomCrop(img_size, padding=4, padding_mode="reflect"),
#         T.RandomHorizontalFlip(),
#         T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
#         T.ToTensor(),
#         T.Normalize(CIFAR_MEAN, CIFAR_STD),
#         T.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
#     ])
#     val = T.Compose([
#         T.ToTensor(),
#         T.Normalize(CIFAR_MEAN, CIFAR_STD),
#     ])
#     return train, val

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
        # T.RandomCrop(img_size, padding=2, padding_mode="edge"),
        T.ToTensor(),
        T.Normalize(FASHION_MEAN, FASHION_STD),
        T.RandomErasing(p=0.1, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
    ])
    val = T.Compose([
        T.ToTensor(),
        T.Normalize(FASHION_MEAN, FASHION_STD),
    ])
    return train, val