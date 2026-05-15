# models/ggm_resnet.py
from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import timm

try:
    from .linearggm import LinearGGM
    from .conv2dggm import Conv2dGGM
    from .custom_resnet import create_resnet20
except ImportError:
    from linearggm import LinearGGM
    from conv2dggm import Conv2dGGM
    from custom_resnet import create_resnet20


def customize_model(
    model: nn.Module,
    N_factor: float,
) -> None:
    
    skip_top = {"conv1", "bn1", "act1", "maxpool", "fc", "head", "stem"}

    for name, module in model.named_children():
        if name in skip_top:
            continue

        replace_layers_recursive(
            parent_module=module,
            N_factor=N_factor,
            current_path=name,
        )


def replace_layers_recursive(
    parent_module: nn.Module,
    N_factor: float,
    current_path: str,
    keep_downsample_fp: bool = True,
) -> None:
    for name, module in parent_module.named_children():
        full_name = f"{current_path}.{name}"

        if keep_downsample_fp and (
            ".downsample" in full_name or full_name.endswith("downsample")
        ):
            continue

        if isinstance(module, nn.Conv2d):
            k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else int(module.kernel_size)
            s = module.stride[0] if isinstance(module.stride, tuple) else int(module.stride)
            p = module.padding[0] if isinstance(module.padding, tuple) else int(module.padding)

            new_conv = Conv2dGGM(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=k,
                stride=s,
                padding=p,
                groups=module.groups,
                N_factor=N_factor,
                bias=(module.bias is not None),
            )

            with torch.no_grad():
                new_conv.weight.copy_(module.weight)
                if module.bias is not None and getattr(new_conv, "bias", None) is not None:
                    new_conv.bias.copy_(module.bias)

            setattr(parent_module, name, new_conv)

        elif isinstance(module, nn.Linear):
            new_linear = LinearGGM(
                in_features=module.in_features,
                out_features=module.out_features,
                N_factor=N_factor,
                bias=(module.bias is not None),
            )
            with torch.no_grad():
                new_linear.weight.copy_(module.weight)
                if module.bias is not None and getattr(new_linear, "bias", None) is not None:
                    new_linear.bias.copy_(module.bias)

            setattr(parent_module, name, new_linear)

        else:
            replace_layers_recursive(
                module,
                N_factor,
                full_name,
                keep_downsample_fp=keep_downsample_fp,
            )


def disable_inplace_activations(model: nn.Module) -> None:

    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False


def set_resnet_prelu(model: nn.Module, init: float = 0.25) -> None:
    if hasattr(model, "bn1") and hasattr(model, "act1"):
        if isinstance(model.bn1, nn.BatchNorm2d) and isinstance(model.act1, nn.Module):
            model.act1 = nn.PReLU(num_parameters=model.bn1.num_features, init=init)

    for m in model.modules():
        if hasattr(m, "bn1") and hasattr(m, "act1") and isinstance(m.bn1, nn.BatchNorm2d):
            m.act1 = nn.PReLU(num_parameters=m.bn1.num_features, init=init)
        if hasattr(m, "bn2") and hasattr(m, "act2") and isinstance(m.bn2, nn.BatchNorm2d):
            m.act2 = nn.PReLU(num_parameters=m.bn2.num_features, init=init)
        if hasattr(m, "bn3") and hasattr(m, "act3") and isinstance(m.bn3, nn.BatchNorm2d):
            m.act3 = nn.PReLU(num_parameters=m.bn3.num_features, init=init)


SMALL_IMAGE_DATASETS = {"cifar10", "cifar100", "fashionmnist"}


def patch_resnet_stem_for_dataset(model: nn.Module, dataset_name: str, in_chans: int) -> None:
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


def _patch_classifier_dropout(model: nn.Module, drop_rate: float) -> None:
    if drop_rate <= 0:
        return

    if hasattr(model, "drop") and isinstance(model.drop, nn.Module) and not isinstance(model.drop, nn.Identity):
        return

    model.drop = nn.Dropout(p=drop_rate)


def build_model(
    model_name: str = "resnet18",
    num_classes: int = 1000,
    N_factor: float = 1.0,
    requires_grad: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    img_size: Optional[int] = None,  
    in_chans: int = 3,
    dataset_name: Optional[str] = None,
    use_prelu: bool = False,      
    prelu_init: float = 0.25,    
) -> nn.Module:

    mn = model_name.lower().strip()

    if mn == "resnet20":
        model = create_resnet20(num_classes=num_classes, in_chans=in_chans)
    else:
        model = timm.create_model(
            model_name,
            pretrained=False, 
            num_classes=num_classes,
            in_chans=in_chans,
        )

    disable_inplace_activations(model)

    if use_prelu:
        set_resnet_prelu(model, init=prelu_init)

    if dataset_name is not None:
        patch_resnet_stem_for_dataset(model, dataset_name, in_chans=in_chans)

    customize_model(model, N_factor=N_factor, )

    for p in model.parameters():
        p.requires_grad = requires_grad

    if device is not None:
        model = model.to(device)

    return model
