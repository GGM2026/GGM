from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from conv2dggd import Conv2dGGD


RESNET_IMAGENET_CONFIGS = {
    "18": dict(block="basic", layers=[2, 2, 2, 2]),
    "34": dict(block="basic", layers=[3, 4, 6, 3]),
    "50": dict(block="bottleneck", layers=[3, 4, 6, 3]),
}


@dataclass
class ResNetImageNetConfig:
    block: str
    layers: list[int]
    num_classes: int = 1000
    in_chans: int = 3
    use_ggd: bool = True
    N_scale: float = 1.0
    prelu_init: float = 0.25
    zero_init_residual: bool = False


def _swap_conv(
    in_planes: int,
    planes: int,
    kernel_size: int,
    stride: int,
    padding: int,
    bias: bool = False,
    use_ggd: bool = True,
    N_scale: float = 1.0,
):
    if use_ggd:
        return Conv2dGGD(
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


class ImageNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        use_ggd: bool = True,
        N_scale: float = 1.0,
    ):
        super().__init__()

        self.conv1 = _swap_conv(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            use_ggd=use_ggd,
            N_scale=N_scale,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = _swap_conv(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            use_ggd=use_ggd,
            N_scale=N_scale,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.act2(out)
        return out


class ImageNetBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        use_ggd: bool = True,
        N_scale: float = 1.0,
    ):
        super().__init__()

        self.conv1 = _swap_conv(
            in_planes,
            planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            use_ggd=use_ggd,
            N_scale=N_scale,
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = _swap_conv(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            use_ggd=use_ggd,
            N_scale=N_scale,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = _swap_conv(
            planes,
            planes * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            use_ggd=use_ggd,
            N_scale=N_scale,
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.act = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.act(out)
        return out


class ResNetImageNet(nn.Module):
    def __init__(self, config: ResNetImageNetConfig):
        super().__init__()

        if config.block == "basic":
            block_cls = ImageNetBasicBlock
        elif config.block == "bottleneck":
            block_cls = ImageNetBottleneck
        else:
            raise ValueError(f"Unknown block type: {config.block}")

        self.in_planes = 64

        # Standard ImageNet stem
        self.conv1 = nn.Conv2d(
            config.in_chans,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block_cls, 64,  config.layers[0], stride=1,
                                       use_ggd=config.use_ggd, N_scale=config.N_scale)
        self.layer2 = self._make_layer(block_cls, 128, config.layers[1], stride=2,
                                       use_ggd=config.use_ggd, N_scale=config.N_scale)
        self.layer3 = self._make_layer(block_cls, 256, config.layers[2], stride=2,
                                       use_ggd=config.use_ggd, N_scale=config.N_scale)
        self.layer4 = self._make_layer(block_cls, 512, config.layers[3], stride=2,
                                       use_ggd=config.use_ggd, N_scale=config.N_scale)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block_cls.expansion, config.num_classes)

    def _make_layer(
        self,
        block_cls,
        planes: int,
        blocks: int,
        stride: int,
        use_ggd: bool,
        N_scale: float,
    ) -> nn.Sequential:
        downsample = None
        out_planes = planes * block_cls.expansion

        if stride != 1 or self.in_planes != out_planes:
            downsample = nn.Sequential(
                _swap_conv(
                    self.in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                    use_ggd=use_ggd,
                    N_scale=N_scale,
                ),
                nn.BatchNorm2d(out_planes),
            )

        layers = [
            block_cls(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                use_ggd=use_ggd,
                N_scale=N_scale,
            )
        ]
        self.in_planes = out_planes

        for _ in range(1, blocks):
            layers.append(
                block_cls(
                    in_planes=self.in_planes,
                    planes=planes,
                    stride=1,
                    downsample=None,
                    use_ggd=use_ggd,
                    N_scale=N_scale,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_resnet_imagenet(
    model_name: str = "18",
    num_classes: int = 1000,
    in_chans: int = 3,
    use_ggd: bool = True,
    N_scale: float = 1.0,
) -> nn.Module:
    if model_name not in RESNET_IMAGENET_CONFIGS:
        raise ValueError(
            f"Unknown model_name: {model_name}. Allowed: {list(RESNET_IMAGENET_CONFIGS)}"
        )

    cfg = ResNetImageNetConfig(
        **RESNET_IMAGENET_CONFIGS[model_name],
        num_classes=num_classes,
        in_chans=in_chans,
        use_ggd=use_ggd,
        N_scale=N_scale,
    )
    return ResNetImageNet(cfg)