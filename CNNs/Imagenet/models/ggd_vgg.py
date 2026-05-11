# models/ggd_vgg.py
'''VGG family (small/medium/large) for CIFAR10. FC layers are removed (ImageNet-style FC stack removed; CIFAR head kept).
(c) YANG, Wei (adapted)
'''
from __future__ import annotations

from typing import Optional, Union
import math

import torch
import torch.nn as nn

from .conv2dggd import Conv2dGGD

class PReLU_plus(nn.Module):
    def __init__(self, num_parameters=1, init_pos=1.0, init_neg=0.25):
        super().__init__()
        self.a_pos = nn.Parameter(torch.full((num_parameters,), init_pos))
        self.a_neg = nn.Parameter(torch.full((num_parameters,), init_neg))

    def forward(self, x):
        # x>=0 -> a_pos*x, x<0 -> a_neg*x
        return torch.where(x >= 0, self.a_pos * x, self.a_neg * x)


# -------------------------
# Backbones
# -------------------------
class VGGSmall(nn.Module):
    """
    CIFAR10-friendly VGG "small":
    - 6 conv layers total (conv0..conv5)
    - 3 max-pools so spatial goes 32->16->8->4
    - Head is Linear(512*4*4 -> num_classes)
    - BN + Hardtanh layout as in your original ggd_vgg_small.py
    """
    def __init__(self, num_classes: int = 10, in_chans: int = 3):
        super().__init__()
        self.conv0 = nn.Conv2d(in_chans, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)

        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(512)

        self.nonlinear = PReLU_plus()

        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)

        self.fc = nn.Linear(512 * 4 * 4, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nonlinear(self.bn0(self.conv0(x)))

        x = self.conv1(x)
        x = self.pooling(x)
        x = self.nonlinear(self.bn1(x))

        x = self.conv2(x)
        x = self.nonlinear(self.bn2(x))

        x = self.conv3(x)
        x = self.pooling(x)
        x = self.nonlinear(self.bn3(x))

        x = self.conv4(x)
        x = self.nonlinear(self.bn4(x))

        x = self.conv5(x)
        x = self.pooling(x)
        x = self.nonlinear(self.bn5(x))

        x = x.view(x.size(0), -1)
        return self.fc(x)


# -------------------------
# VGG-11 and VGG-16 Build
# -------------------------

_CFG = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],  # VGG11
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512],  # VGG16
}

def make_layers(cfg, in_channels: int = 3, batch_norm: bool = True, dropout2d_p: float = 0.1) -> nn.Sequential:
    layers = []
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.PReLU(v)]
            else:
                layers += [conv2d, nn.PReLU(v)]

            if dropout2d_p > 0:
                layers.append(nn.Dropout2d(p=dropout2d_p))

            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    """
    CIFAR VGG wrapper: features (conv/bn/act/pool) + linear head.
    Assumes 5 pools -> spatial ends at 1x1, so feature dim is 512.
    """
    def __init__(self, features: nn.Sequential, num_classes: int = 10, dropout_p: float = 0.3):
        super().__init__()
        self.features = features
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(512 * 2 * 2, num_classes)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)  # N x 512
        x = self.dropout(x)
        return self.fc(x)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
class VGG11(VGG):
    def __init__(self, num_classes: int = 10, in_chans: int = 3):
        super().__init__(features=make_layers(_CFG["A"], in_channels=in_chans, batch_norm=True),
                         num_classes=num_classes)

class VGG16(VGG):
    def __init__(self, num_classes: int = 10, in_chans: int = 3):
        super().__init__(features=make_layers(_CFG["D"], in_channels=in_chans, batch_norm=True),
                         num_classes=num_classes)


# -------------------------
# Replace layers with GGD
# -------------------------
def _swap_conv_to_ggd(conv: nn.Conv2d, N_scale: float,) -> Conv2dGGD:
    k = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else int(conv.kernel_size)
    s = conv.stride[0] if isinstance(conv.stride, tuple) else int(conv.stride)
    p = conv.padding[0] if isinstance(conv.padding, tuple) else int(conv.padding)

    new_conv = Conv2dGGD(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=k,
        stride=s,
        padding=p,
        groups=conv.groups,
        N_scale=N_scale,
        bias=(conv.bias is not None),
    )

    with torch.no_grad():
        new_conv.weight.copy_(conv.weight)
        if conv.bias is not None and getattr(new_conv, "bias", None) is not None:
            new_conv.bias.copy_(conv.bias)
        if hasattr(new_conv, "scale"):
            new_conv.scale.data.fill_(1.0)

    return new_conv


def customize_model(model: nn.Module, N_scale: float,) -> None:
    """
    Replace all conv layers except the stem conv with Conv2dGGD.
    Supports:
      - VGGSmall with explicit conv0/conv1/...
      - VGG{11,16} with model.features = nn.Sequential(...)
    """
    # Identify the stem conv module object to skip.
    stem_conv = None
    if hasattr(model, "conv0") and isinstance(getattr(model, "conv0"), nn.Conv2d):
        stem_conv = getattr(model, "conv0")
    elif hasattr(model, "features") and isinstance(getattr(model, "features"), nn.Sequential):
        # In our make_layers(batch_norm=True), features[0] is the first conv
        if len(model.features) > 0 and isinstance(model.features[0], nn.Conv2d):
            stem_conv = model.features[0]

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if stem_conv is not None and module is stem_conv:
            continue

        # Find parent and replace
        parent = model
        if "." in module_name:
            *parent_path, child_name = module_name.split(".")
            for p in parent_path:
                parent = getattr(parent, p)
        else:
            child_name = module_name

        setattr(parent, child_name, _swap_conv_to_ggd(module, N_scale=N_scale,))


def disable_inplace_activations(model: nn.Module) -> None:
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False


# -------------------------
# Build
# -------------------------
_VGG_VARIANTS = {
    "vgg_small": (VGGSmall, "small"),
    "vgg11": (VGG11, "vgg11"),
    "vgg16": (VGG16, "vgg16"),

    "vgg_small_1w1a": (VGGSmall, "small"),
    "vgg11_1w1a": (VGG11, "vgg11"),
    "vgg16_1w1a": (VGG16, "vgg16"),
}




def build_model(
    model_name: str = "vgg16",
    num_classes: int = 10,
    N_scale: float = 1.0,
    requires_grad: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    img_size: Optional[int] = None,  # unused; kept for API compatibility
    in_chans: int = 3,
    dataset_name: Optional[str] = None,  # unused; kept for API compatibility
) -> nn.Module:
    """
    Factory to build and customize VGG family for CIFAR10 with Conv2d swapped to Conv2dGGD.
    
    model_name:
      - "vgg_small", "vgg11", "vgg16"
      - plus "*_1w1a" aliases
    """
    mn = model_name.lower().strip()
    if mn not in _VGG_VARIANTS:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Choose from: {sorted(_VGG_VARIANTS.keys())}"
        )

    model_cls, _size = _VGG_VARIANTS[mn]

    # 1) Build full-precision backbone
    model = model_cls(num_classes=num_classes, in_chans=in_chans)

    disable_inplace_activations(model)

    # 2) Always customize (monkeypatch in __init__.py will bypass this for FP runs)
    customize_model(model, N_scale=N_scale,)

    for p in model.parameters():
        p.requires_grad = requires_grad

    if device is not None:
        model = model.to(device)

    return model
