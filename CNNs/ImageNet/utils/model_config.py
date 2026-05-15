# utils/model_config.py
from __future__ import annotations


def resolve_timm_backbone(model_family: str, size: str) -> str:
    fam = model_family.lower().strip()
    sz = str(size).lower().strip()

    if fam != "resnet":
        raise ValueError(f"Only resnet is supported currently, got {fam}")

    allowed = {"18", "20", "34"}
    if sz not in allowed:
        raise ValueError(f"Invalid --size '{size}'. Allowed: {sorted(allowed)}")

    if sz == "20":
        return "resnet20" 
    return f"resnet{sz}"


def validate_global_model_dataset_args(args) -> None:
    if args.model.lower() == "resnet" and str(args.size) == "20" and args.dataset.lower() == "imagenet":
        raise ValueError(
            "Invalid config: --model resnet --size 20 is a CIFAR-style ResNet-20 and is not supported for --dataset imagenet. "
            "Use --size 18, 34 for ImageNet, or switch dataset to cifar10/cifar100/fashionmnist."
        )
