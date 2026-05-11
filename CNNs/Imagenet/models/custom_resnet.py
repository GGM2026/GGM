# models/custom_resnet20.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFARBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        self.act2 = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.act2(out)
        return out


class ResNetCIFAR(nn.Module):
    """
    CIFAR-style ResNet-20:
      - 3x3 conv stem, stride 1
      - no maxpool
      - stages: 16, 32, 64 channels
      - depth=20 corresponds to n=3 blocks per stage
    Exposes conv1/bn1/act1/maxpool/fc so your GGD code can skip top-level stem/head.
    """
    def __init__(self, num_classes: int = 10, in_chans: int = 3):
        super().__init__()
        self.in_planes = 16

        # Keep naming compatible with your ggd_resnet.py expectations
        self.conv1 = nn.Conv2d(in_chans, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU(inplace=False)
        self.maxpool = nn.Identity()  # exists for compatibility

        # ResNet-20 => 3 blocks per stage
        self.layer1 = self._make_layer(16, blocks=3, stride=1)
        self.layer2 = self._make_layer(32, blocks=3, stride=2)
        self.layer3 = self._make_layer(64, blocks=3, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [CIFARBasicBlock(self.in_planes, planes, stride=stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(CIFARBasicBlock(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def create_resnet20(num_classes: int, in_chans: int) -> nn.Module:
    """Factory for CIFAR-style ResNet-20."""
    return ResNetCIFAR(num_classes=num_classes, in_chans=in_chans)
