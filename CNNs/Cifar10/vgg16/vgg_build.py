from __future__ import annotations

from dataclasses import dataclass
from typing import Union
import math

import torch
import torch.nn as nn

from conv2dggd import Conv2dGGD


VGG_CONFIGS = {
    "small": dict(cfg=None),
    "11": dict(cfg=[64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]),
    "16": dict(cfg=[64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512]),
}


@dataclass
class VGGConfig:
    variant: str
    num_classes: int = 10
    in_chans: int = 3
    use_ggd: bool = True
    N_scale: float = 1.0
    prelu_init: float = 0.25
    dropout2d_p: float = 0.1
    dropout_p: float = 0.3


class PReLUPlus(nn.Module):
    def __init__(self, num_parameters: int = 1, init_pos: float = 1.0, init_neg: float = 0.25):
        super().__init__()
        self.a_pos = nn.Parameter(torch.full((num_parameters,), init_pos))
        self.a_neg = nn.Parameter(torch.full((num_parameters,), init_neg))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, self.a_pos * x, self.a_neg * x)


def _make_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    bias: bool = False,
    use_ggd: bool = True,
    N_scale: float = 1.0,
    *,
    is_stem: bool = False,
):
    if use_ggd and not is_stem:
        return Conv2dGGD(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            N_scale=N_scale,
        )

    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )


class VGGSmall(nn.Module):
    def __init__(self, config: VGGConfig):
        super().__init__()

        self.conv0 = _make_conv(
            config.in_chans,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            use_ggd=config.use_ggd,
            N_scale=config.N_scale,
            is_stem=True,
        )
        self.bn0 = nn.BatchNorm2d(128)
        self.act0 = PReLUPlus(init_neg=config.prelu_init)

        self.conv1 = _make_conv(128, 128, 3, 1, 1, bias=False, use_ggd=config.use_ggd, N_scale=config.N_scale)
        self.bn1 = nn.BatchNorm2d(128)
        self.act1 = PReLUPlus(init_neg=config.prelu_init)

        self.conv2 = _make_conv(128, 256, 3, 1, 1, bias=False, use_ggd=config.use_ggd, N_scale=config.N_scale)
        self.bn2 = nn.BatchNorm2d(256)
        self.act2 = PReLUPlus(init_neg=config.prelu_init)

        self.conv3 = _make_conv(256, 256, 3, 1, 1, bias=False, use_ggd=config.use_ggd, N_scale=config.N_scale)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = PReLUPlus(init_neg=config.prelu_init)

        self.conv4 = _make_conv(256, 512, 3, 1, 1, bias=False, use_ggd=config.use_ggd, N_scale=config.N_scale)
        self.bn4 = nn.BatchNorm2d(512)
        self.act4 = PReLUPlus(init_neg=config.prelu_init)

        self.conv5 = _make_conv(512, 512, 3, 1, 1, bias=False, use_ggd=config.use_ggd, N_scale=config.N_scale)
        self.bn5 = nn.BatchNorm2d(512)
        self.act5 = PReLUPlus(init_neg=config.prelu_init)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(512 * 4 * 4, config.num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, Conv2dGGD)):
                if hasattr(m, "weight") and m.weight is not None:
                    k = m.kernel_size
                    if isinstance(k, tuple):
                        kh, kw = k
                    else:
                        kh = kw = int(k)
                    n = kh * kw * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.data.zero_()
                if isinstance(m, Conv2dGGD) and hasattr(m, "scale"):
                    m.scale.data.fill_(1.0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act0(self.bn0(self.conv0(x)))

        x = self.conv1(x)
        x = self.pool(x)
        x = self.act1(self.bn1(x))

        x = self.conv2(x)
        x = self.act2(self.bn2(x))

        x = self.conv3(x)
        x = self.pool(x)
        x = self.act3(self.bn3(x))

        x = self.conv4(x)
        x = self.act4(self.bn4(x))

        x = self.conv5(x)
        x = self.pool(x)
        x = self.act5(self.bn5(x))

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class VGGGM(nn.Module):
    def __init__(self, config: VGGConfig):
        super().__init__()

        cfg = VGG_CONFIGS[config.variant]["cfg"]
        self.features = self._make_layers(cfg, config)
        self.dropout = nn.Dropout(p=config.dropout_p)
        self.fc = nn.Linear(512 * 2 * 2, config.num_classes)

        self._initialize_weights()

    def _make_layers(self, cfg: list[Union[int, str]], config: VGGConfig) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_channels = config.in_chans
        is_stem = True

        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue

            conv = _make_conv(
                in_channels=in_channels,
                out_channels=v,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                use_ggd=config.use_ggd,
                N_scale=config.N_scale,
                is_stem=is_stem,
            )
            layers.extend([
                conv,
                nn.BatchNorm2d(v),
                nn.PReLU(num_parameters=v, init=config.prelu_init),
            ])

            if config.dropout2d_p > 0:
                layers.append(nn.Dropout2d(p=config.dropout2d_p))

            in_channels = v
            is_stem = False

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, Conv2dGGD)):
                if hasattr(m, "weight") and m.weight is not None:
                    k = m.kernel_size
                    if isinstance(k, tuple):
                        kh, kw = k
                    else:
                        kh = kw = int(k)
                    n = kh * kw * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.data.zero_()
                if isinstance(m, Conv2dGGD) and hasattr(m, "scale"):
                    m.scale.data.fill_(1.0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def build_vgg_cifar(
    model_name: str = "16",
    num_classes: int = 10,
    in_chans: int = 3,
    use_ggd: bool = True,
    N_scale: float = 1.0,
    prelu_init: float = 0.25,
    dropout2d_p: float = 0.1,
    dropout_p: float = 0.3,
) -> nn.Module:
    aliases = {
        "vgg_small": "small",
        "vgg11": "11",
        "vgg16": "16",
        "small": "small",
        "11": "11",
        "16": "16",
        "vgg_small_1w1a": "small",
        "vgg11_1w1a": "11",
        "vgg16_1w1a": "16",
    }

    key = aliases.get(model_name.lower().strip())
    if key is None:
        raise ValueError(f"Unknown model_name: {model_name}. Allowed: {sorted(aliases)}")

    config = VGGConfig(
        variant=key,
        num_classes=num_classes,
        in_chans=in_chans,
        use_ggd=use_ggd,
        N_scale=N_scale,
        prelu_init=prelu_init,
        dropout2d_p=dropout2d_p,
        dropout_p=dropout_p,
    )

    if key == "small":
        return VGGSmall(config)
    return VGGGM(config)


def build_model(
    model_name: str = "vgg16",
    num_classes: int = 10,
    N_scale: float = 1.0,
    requires_grad: bool = True,
    device=None,
    img_size=None,
    in_chans: int = 3,
    dataset_name=None,
    use_ggd: bool = True,
    prelu_init: float = 0.25,
    dropout2d_p: float = 0.1,
    dropout_p: float = 0.3,
) -> nn.Module:
    del img_size, dataset_name

    model = build_vgg_cifar(
        model_name=model_name,
        num_classes=num_classes,
        in_chans=in_chans,
        use_ggd=use_ggd,
        N_scale=N_scale,
        prelu_init=prelu_init,
        dropout2d_p=dropout2d_p,
        dropout_p=dropout_p,
    )

    for p in model.parameters():
        p.requires_grad = requires_grad

    if device is not None:
        model = model.to(device)

    return model
