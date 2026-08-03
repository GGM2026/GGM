import random
import numpy as np
import torch
import torch.nn as nn
import math

def make_G_from_seed(seed: int, N: int, d: int, device):
    import math
    import torch

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    blocks = []
    remaining = int(N)

    while remaining > 0:
        A = torch.randn(d, d, generator=g, device="cpu")
        Q, R = torch.linalg.qr(A)

        signs = torch.sign(torch.diag(R))
        signs[signs == 0] = 1.0
        Q = Q * signs.view(1, -1)

        take = min(remaining, d)
        blocks.append(Q[:take])
        remaining -= take

    G = torch.cat(blocks, dim=0)
    G = G * math.sqrt(d)

    return G.to(device)

class LayerScale(nn.Module):
    def __init__(self, hidden_size: int, init_value: float = 0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(hidden_size) * init_value)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.alpha + self.bias

class OddGate(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, z):
        return z * torch.tanh(self.alpha * z)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        orig_dtype = x.dtype

        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        y = x_fp32 / rms * self.weight

        return y.to(orig_dtype)

class RPReLU(nn.Module):
    """
    RPReLU(x) = PReLU(x - gamma) + zeta

    gamma: learnable input shift
    zeta:  learnable output shift
    alpha: learnable negative slope
    """
    def __init__(self, num_channels, channel_dim=-1, slope_init=0.25):
        super().__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim

        self.gamma = nn.Parameter(torch.zeros(num_channels))
        self.zeta = nn.Parameter(torch.zeros(num_channels))
        self.alpha = nn.Parameter(torch.ones(num_channels) * slope_init)

    def _view(self, p, x):
        dim = self.channel_dim
        if dim < 0:
            dim = x.ndim + dim

        shape = [1] * x.ndim
        shape[dim] = self.num_channels
        return p.view(*shape)

    def forward(self, x):
        gamma = self._view(self.gamma, x)
        zeta = self._view(self.zeta, x)
        alpha = self._view(self.alpha, x)

        y = x - gamma
        y = torch.where(y >= 0, y, alpha * y)
        return y + zeta