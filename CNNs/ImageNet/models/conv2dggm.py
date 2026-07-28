# conv2dggm.py

import math
import random
from typing import Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Seed utility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# QR-based G generation
# ============================================================

def make_G_from_seed(seed: int, N: int, d: int, device):
    """
    Generate G with QR-orthogonal rows.

    For N <= d:
        rows are orthogonal.

    For N > d:
        rows are generated in repeated d x d orthogonal blocks.

    The final rows are scaled by sqrt(d) so that row norms match
    the typical norm scale of torch.randn(N, d).
    """
    seed = int(seed)
    N = int(N)
    d = int(d)

    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    blocks = []
    remaining = N

    while remaining > 0:
        A = torch.randn(d, d, generator=gen, device="cpu")
        Q, R = torch.linalg.qr(A)

        # Make the QR sign convention deterministic.
        signs = torch.sign(torch.diag(R))
        signs[signs == 0] = 1.0
        Q = Q * signs.view(1, -1)

        take = min(remaining, d)
        blocks.append(Q[:take])
        remaining -= take

    G = torch.cat(blocks, dim=0)
    G = G * math.sqrt(d)

    return G.to(device)


def make_G_conv2d_from_seed(
    seed: int,
    groups: int,
    N: int,
    cin_g: int,
    kh: int,
    kw: int,
    device,
):
    """
    Build grouped convolutional G.

    Returns:
        G with shape [groups, N, cin_g, kh, kw]
    """
    groups = int(groups)
    N = int(N)
    cin_g = int(cin_g)
    kh = int(kh)
    kw = int(kw)

    d = cin_g * kh * kw

    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")

    group_blocks = []

    for group_idx in range(groups):
        group_seed = int(seed + 10007 * group_idx)

        Gg = make_G_from_seed(
            seed=group_seed,
            N=N,
            d=d,
            device=torch.device("cpu"),
        )

        Gg = Gg.reshape(N, cin_g, kh, kw)
        group_blocks.append(Gg)

    G = torch.stack(group_blocks, dim=0)
    return G.to(device)


# ============================================================
# Small utility
# ============================================================

def _pair(v):
    if isinstance(v, int):
        return (v, v)
    if isinstance(v, tuple):
        return v
    return tuple(v)


# ============================================================
# Conv2dGGM
# ============================================================

class Conv2dGGM(nn.Module):
    """
    Conv2d replacement using a binary GGM random-feature projection.

    Forward:
        - Project input patches with fixed QR-generated G.
        - Binarize projected activations.
        - Project weights with the same G.
        - Binarize projected weights.
        - Estimate the kernel via averaged binary products.

    Backward:
        - Straight-through estimator using the analytic arcsin surrogate.

    G shape:
        [groups, N, cin_g, kh, kw]

    Activation projection shape:
        G -> [groups * N, cin_g, kh, kw]

    Weight projection shape:
        G -> [groups, N, cin_g * kh * kw]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        N_factor: float = 2.5,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = False,
        eps: float = 1e-6,
        chunk_N: int = 0,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        kh, kw = _pair(kernel_size)
        self.kh = int(kh)
        self.kw = int(kw)

        # Preserve old attribute name for compatibility.
        self.kernel_size = self.kh if self.kh == self.kw else (self.kh, self.kw)

        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.groups = int(groups)
        self.eps = float(eps)
        self.chunk_N = int(chunk_N)

        if self.groups <= 0:
            raise ValueError(f"groups must be positive, got {self.groups}")
        if self.in_channels % self.groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if self.out_channels % self.groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.cin_g = self.in_channels // self.groups
        self.cout_g = self.out_channels // self.groups

        K = self.cin_g * self.kh * self.kw
        self.K = int(K)

        self.N_factor = float(N_factor)
        self.N = int(self.K * self.N_factor)

        self.weight = nn.Parameter(
            torch.randn(
                self.out_channels,
                self.cin_g,
                self.kh,
                self.kw,
            ) * 0.05
        )

        self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None

        # Generate G once on CPU, then register as a buffer.
        # model.to(device) will move this buffer automatically.
        self.G_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        G = make_G_conv2d_from_seed(
            seed=self.G_seed,
            groups=self.groups,
            N=self.N,
            cin_g=self.cin_g,
            kh=self.kh,
            kw=self.kw,
            device=torch.device("cpu"),
        )

        self.register_buffer("G", G)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.groups
        N = self.N
        cin_g = self.cin_g
        cout_g = self.cout_g
        kh = self.kh
        kw = self.kw
        K = self.K

        x = x - x.mean(dim=(1, 2, 3), keepdim=True)

        # ----------------------------------------------------
        # Binary forward path
        # ----------------------------------------------------

        # Activation projection:
        Gx = self.G.to(device=x.device, dtype=x.dtype)
        G_conv = Gx.reshape(g * N, cin_g, kh, kw)

        z = F.conv2d(
            x,
            G_conv,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            groups=g,
        )

        z_b = torch.where(
            z >= 0,
            torch.ones((), device=z.device, dtype=z.dtype),
            -torch.ones((), device=z.device, dtype=z.dtype),
        )

        # Weight projection:
        W_flat = self.weight.reshape(g, cout_g, K)

        Gw = self.G.to(device=x.device, dtype=W_flat.dtype)
        Gw = Gw.reshape(g, N, K)

        W_proj = torch.bmm(
            Gw,
            W_flat.transpose(1, 2),
        )  # [g, N, cout_g]

        W_b = torch.where(
            W_proj >= 0,
            torch.ones((), device=W_proj.device, dtype=W_proj.dtype),
            -torch.ones((), device=W_proj.device, dtype=W_proj.dtype),
        )

        W_b = W_b.transpose(1, 2).reshape(self.out_channels, N, 1, 1)

        y_bin = F.conv2d(
            z_b,
            W_b.to(dtype=z_b.dtype),
            bias=None,
            stride=1,
            padding=0,
            groups=g,
        ) / float(N)

        # ----------------------------------------------------
        # Smooth surrogate path for gradients
        # ----------------------------------------------------

        x32 = x.float()
        W32 = self.weight.float()

        num = F.conv2d(
            x32,
            W32,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            groups=g,
        )

        ones = torch.ones(
            g,
            cin_g,
            kh,
            kw,
            device=x.device,
            dtype=x32.dtype,
        )

        p2 = F.conv2d(
            x32.square(),
            ones,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            groups=g,
        )

        inv_pn = torch.rsqrt(p2 + self.eps)
        inv_pn = inv_pn.repeat_interleave(cout_g, dim=1)

        inv_wn = torch.rsqrt(
            W32.square().sum(dim=(1, 2, 3)) + self.eps
        ).reshape(1, self.out_channels, 1, 1)

        corr = num * inv_pn * inv_wn

        y_surr = (2.0 / math.pi) * torch.asin(corr)
        y_surr = y_surr.to(dtype=y_bin.dtype)

        # Straight-through estimator:
        #   forward value comes from y_bin,
        #   backward gradient comes from y_surr.
        y = y_bin.detach() + (y_surr - y_surr.detach())


        if self.bias is not None:
            y = y + self.bias.reshape(1, -1, 1, 1)

        return y

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size=({self.kh}, {self.kw}), "
            f"N={self.N}, "
            f"N_factor={self.N_factor}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"groups={self.groups}, "
            f"bias={self.bias is not None}, "
            f"G_shape={tuple(self.G.shape)}"
        )