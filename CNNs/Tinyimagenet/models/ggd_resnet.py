from __future__ import annotations

from typing import Optional, Union

import math
import torch
import torch.nn as nn
import timm

from .linearggd import LinearGGD
from .conv2dggd import Conv2dGGD
from .custom_resnet import create_resnet20


def customize_model(
    model: nn.Module,
    N_scale: float,
    chunk_N: int = 0,
) -> None:
    """
    Replace Conv2d/Linear layers with Conv2dGGD/LinearGGD.
    Customizes everything except the top-level stem and head.
    """
    skip_top = {"conv1", "bn1", "act1", "maxpool", "fc", "head", "stem"}

    for name, module in model.named_children():
        if name in skip_top:
            continue

        replace_layers_recursive(
            parent_module=module,
            N_scale=N_scale,
            current_path=name,
            chunk_N=chunk_N,
        )


def replace_layers_recursive(
    parent_module: nn.Module,
    N_scale: float,
    current_path: str,
    chunk_N: int = 0,
) -> None:
    for name, module in parent_module.named_children():
        full_name = f"{current_path}.{name}"

        if isinstance(module, nn.Conv2d):
            k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else int(module.kernel_size)
            s = module.stride[0] if isinstance(module.stride, tuple) else int(module.stride)
            p = module.padding[0] if isinstance(module.padding, tuple) else int(module.padding)
            
            new_conv = Conv2dGGD(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=k,
                stride=s,
                padding=p,
                groups=module.groups,
                N_scale=N_scale,
                bias=(module.bias is not None),
                chunk_N=chunk_N,
            )

            with torch.no_grad():
                new_conv.weight.copy_(module.weight)
                if module.bias is not None and getattr(new_conv, "bias", None) is not None:
                    new_conv.bias.copy_(module.bias)

            setattr(parent_module, name, new_conv)

        elif isinstance(module, nn.Linear):
            new_linear = LinearGGD(
                in_features=module.in_features,
                out_features=module.out_features,
                N_scale=N_scale,
                bias=(module.bias is not None),
            )
            with torch.no_grad():
                new_linear.weight.copy_(module.weight)
                if module.bias is not None and getattr(new_linear, "bias", None) is not None:
                    new_linear.bias.copy_(module.bias)

            setattr(parent_module, name, new_linear)

        else:
            replace_layers_recursive(module, N_scale, full_name, chunk_N=chunk_N)


def disable_inplace_activations(model: nn.Module) -> None:
    """
    ResNet uses inplace=True ReLU by default (timm).
    Ensure inplace=False to avoid issues in some custom layers/backward passes.
    """
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False


SMALL_IMAGE_DATASETS = {"cifar10", "cifar100", "fashionmnist","tinyimagenet"}


def patch_resnet_stem_for_dataset(model: nn.Module, dataset_name: str, in_chans: int) -> None:
    """
    For timm ResNets: swap 7x7/stride2 stem -> 3x3/stride1 and remove maxpool on small-image datasets.

    For custom ResNet20: it already uses a 3x3 stem and Identity maxpool, so this is effectively harmless.
    """
    d = dataset_name.lower()
    if d not in SMALL_IMAGE_DATASETS:
        return

    if not hasattr(model, "conv1"):
        return

    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=in_chans,
        out_channels=old_conv.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.conv1 = new_conv

    if hasattr(model, "maxpool"):
        model.maxpool = nn.Identity()


def build_model(
    model_name: str = "resnet18",
    num_classes: int = 1000,
    N_scale: float = 1.0,
    requires_grad: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    img_size: Optional[int] = None,
    in_chans: int = 3,
    dataset_name: Optional[str] = None,
    chunk_N: int = 0,
    drop_path_rate: float = 0.0,
    full_precision: bool = False,
) -> nn.Module:
    """
    Factory to build and customize ResNet with Conv/Linear swapped to GGD.

    model_name:
      - "resnet18", "resnet34", "resnet50", -----> timm.create_model
      - "resnet20" ------------------------------> custom CIFAR-style ResNet-20
    """
    mn = model_name.lower().strip()

    if mn == "resnet20":
        model = create_resnet20(num_classes=num_classes, in_chans=in_chans, width_mult=1)
    else:
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            in_chans=in_chans,
        )

    disable_inplace_activations(model)

    if not full_precision:
        customize_model(model, N_scale=N_scale, chunk_N=chunk_N)
        
    for p in model.parameters():
        p.requires_grad = requires_grad

    if device is not None:
        model = model.to(device)

    return model
