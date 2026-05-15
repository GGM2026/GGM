from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from conv2dggm import Conv2dGGM


RESNET_CONFIGS = {
    "18": dict(layers=[2, 2, 2, 2], channels=[64, 128, 256, 512]),
    "20": dict(layers=[3, 3, 3], channels=[16, 32, 64]),
}


@dataclass
class ResNetConfig:
    layers: list[int]
    channels: list[int]
    num_classes: int = 10
    in_chans: int = 3
    use_ggm: bool = True
    N_scale: float = 1.0
    prelu_init: float = 0.25


def _swap_conv(
    in_planes: int,
    planes: int,
    kernel_size: int,
    stride: int,
    padding: int,
    bias: bool = True,
    use_ggm: bool = True,
    N_scale: float = 1.0,
):
    if use_ggm:
        return Conv2dGGM(
            in_planes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            N_scale=N_scale,
        )

    return nn.Conv2d(
        in_planes,
        planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )


class CIFARBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        use_ggm: bool = True,
        N_scale: float = 1.0,
        prelu_init: float = 0.25,
    ):
        super().__init__()

        self.conv1 = _swap_conv(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            use_ggm=use_ggm,
            N_scale=N_scale,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.PReLU(num_parameters=planes, init=prelu_init)

        self.conv2 = _swap_conv(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            use_ggm=use_ggm,
            N_scale=N_scale,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.PReLU(num_parameters=planes, init=prelu_init)

        self.downsample = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                _swap_conv(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                    use_ggm=use_ggm,
                    N_scale=N_scale,
                ),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.act2(out)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(self, config: ResNetConfig):
        super().__init__()

        if len(config.layers) != len(config.channels):
            raise ValueError("layers and channels must have the same length")

        self.layers = config.layers
        self.channels = config.channels
        self.in_planes = config.channels[0]

        self.conv1 = nn.Conv2d(
            config.in_chans,
            self.in_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.act1 = nn.PReLU(num_parameters=self.in_planes, init=config.prelu_init)
        self.maxpool = nn.Identity()

        stage_names = []
        for i, (num_blocks, planes) in enumerate(zip(self.layers, self.channels), start=1):
            stride = 1 if i == 1 else 2
            layer = self._make_layer(
                planes=planes,
                blocks=num_blocks,
                stride=stride,
                use_ggm=config.use_ggm,
                N_scale=config.N_scale,
                prelu_init=config.prelu_init,
            )
            name = f"layer{i}"
            setattr(self, name, layer)
            stage_names.append(name)

        self.stage_names = stage_names
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.channels[-1], config.num_classes)

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int,
        use_ggm: bool,
        N_scale: float,
        prelu_init: float,
    ) -> nn.Sequential:
        layers = [
            CIFARBasicBlock(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                use_ggm=use_ggm,
                N_scale=N_scale,
                prelu_init=prelu_init,
            )
        ]
        self.in_planes = planes

        for _ in range(1, blocks):
            layers.append(
                CIFARBasicBlock(
                    in_planes=self.in_planes,
                    planes=planes,
                    stride=1,
                    use_ggm=use_ggm,
                    N_scale=N_scale,
                    prelu_init=prelu_init,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        for name in self.stage_names:
            x = getattr(self, name)(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_resnet_cifar(
    model_name: str = "18",
    num_classes: int = 10,
    in_chans: int = 3,
    use_ggm: bool = True,
    N_scale: float = 1.0,
    prelu_init: float = 0.25,
) -> nn.Module:
    if model_name not in RESNET_CONFIGS:
        raise ValueError(f"Unknown model_name: {model_name}. Allowed: {list(RESNET_CONFIGS)}")

    cfg = ResNetConfig(
        **RESNET_CONFIGS[model_name],
        num_classes=num_classes,
        in_chans=in_chans,
        use_ggm=use_ggm,
        N_scale=N_scale,
        prelu_init=prelu_init,
    )
    return ResNetCIFAR(cfg)