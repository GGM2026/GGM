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
    """
    Checks that should run once early in main(), before building loaders/models.
    """
    if args.model.lower() == "resnet" and str(args.size) == "20" and args.dataset.lower() == "imagenet":
        raise ValueError(
            "Invalid config: --model resnet --size 20 is a CIFAR-style ResNet-20 and is not supported for --dataset imagenet. "
            "Use --size 18, 34 for ImageNet, or switch dataset to cifar10/cifar100/fashionmnist."
        )


def validate_vgg_args(args) -> None:
    """
    VGG-small/medium currently assume CIFAR (32x32) due to fixed FC shape.
    Call this right before creating vgg_small/vgg_medium.
    """
    if args.dataset.lower() not in ("cifar10", "cifar100"):
        raise ValueError(f"{args.model} is only supported for CIFAR10/CIFAR100 (32x32).")

    if args.img_size != 32:
        raise ValueError(f"{args.model} expects --img_size 32 (fixed fc=512*4*4).")
