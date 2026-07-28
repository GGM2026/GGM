# conv1dggd_new.py

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
    """
    Seed Python, NumPy, PyTorch CPU, and PyTorch CUDA RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# QR-based G generation
# ============================================================

def make_G_from_seed(
    seed: int,
    N: int,
    d: int,
    device,
):
    """
    Generate G with QR-orthogonal rows.

    For N <= d:
        All returned rows are mutually orthogonal.

    For N > d:
        Rows are generated in repeated d x d orthogonal blocks.
        Rows within each block are mutually orthogonal.

    The rows are scaled by sqrt(d), giving every row norm sqrt(d).
    This matches the typical norm scale of a d-dimensional standard
    Gaussian vector.

    Args:
        seed:
            Random seed used by a CPU torch.Generator.

        N:
            Number of projection vectors.

        d:
            Dimension of every projection vector.

        device:
            Device to which the finished matrix is moved.

    Returns:
        Tensor with shape [N, d].
    """
    seed = int(seed)
    N = int(N)
    d = int(d)

    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")

    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")

    # Generate on CPU for deterministic behavior across devices.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    blocks = []
    remaining = N

    while remaining > 0:
        # Generate a square Gaussian matrix.
        A = torch.randn(
            d,
            d,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )

        # For a square full-rank matrix, Q is an orthogonal matrix.
        Q, R = torch.linalg.qr(A)

        # QR is ambiguous up to column-wise signs. Normalize the signs
        # using the diagonal of R so the convention is deterministic.
        signs = torch.sign(torch.diag(R))
        signs[signs == 0] = 1.0

        Q = Q * signs.view(1, -1)

        # Since Q is square and orthogonal, both its rows and columns
        # form orthonormal systems.
        take = min(remaining, d)
        blocks.append(Q[:take])

        remaining -= take

    # Shape: [N, d]
    G = torch.cat(blocks, dim=0)

    # Every row of Q has unit norm. Scale it to norm sqrt(d).
    G = G * math.sqrt(d)

    return G.to(device)


def make_G_conv1d_from_seed(
    seed: int,
    groups: int,
    N: int,
    cin_g: int,
    kernel_size: int,
    device,
):
    """
    Build the grouped 1D convolution projection tensor.

    Each convolution group receives its own independently seeded
    QR projection matrix.

    Args:
        seed:
            Base seed.

        groups:
            Number of convolution groups.

        N:
            Number of random projection filters per group.

        cin_g:
            Number of input channels per group.

        kernel_size:
            Length of each 1D convolution kernel.

        device:
            Device to which the finished tensor is moved.

    Returns:
        G with shape:

            [groups, N, cin_g, kernel_size]
    """
    seed = int(seed)
    groups = int(groups)
    N = int(N)
    cin_g = int(cin_g)
    kernel_size = int(kernel_size)

    if groups <= 0:
        raise ValueError(
            f"groups must be positive, got {groups}"
        )

    if N <= 0:
        raise ValueError(
            f"N must be positive, got {N}"
        )

    if cin_g <= 0:
        raise ValueError(
            f"cin_g must be positive, got {cin_g}"
        )

    if kernel_size <= 0:
        raise ValueError(
            f"kernel_size must be positive, got {kernel_size}"
        )

    # Dimension of one flattened grouped convolution kernel.
    d = cin_g * kernel_size

    group_blocks = []

    for group_idx in range(groups):
        # Give every convolution group a deterministic but distinct seed.
        group_seed = seed + 10007 * group_idx

        # Shape: [N, cin_g * kernel_size]
        Gg = make_G_from_seed(
            seed=group_seed,
            N=N,
            d=d,
            device=torch.device("cpu"),
        )

        # Interpret each projection vector as a 1D convolution kernel.
        #
        # Shape: [N, cin_g, kernel_size]
        Gg = Gg.reshape(
            N,
            cin_g,
            kernel_size,
        )

        group_blocks.append(Gg)

    # Shape: [groups, N, cin_g, kernel_size]
    G = torch.stack(
        group_blocks,
        dim=0,
    )

    return G.to(device)


# ============================================================
# Small utility
# ============================================================

def _single(v):
    """
    Normalize a Conv1d scalar argument.

    Conv1d normally uses integer kernel size, stride, padding, and
    dilation values. A length-one tuple is also accepted here.
    """
    if isinstance(v, int):
        return int(v)

    if isinstance(v, tuple):
        if len(v) != 1:
            raise ValueError(
                f"Expected an int or a length-one tuple, got {v}"
            )
        return int(v[0])

    values = tuple(v)

    if len(values) != 1:
        raise ValueError(
            f"Expected an int or a length-one sequence, got {values}"
        )

    return int(values[0])


# ============================================================
# Conv1dGGD
# ============================================================

class Conv1dGGM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        N_factor: float = 2.5,
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: bool = False,
        eps: float = 1e-6,
        chunk_N: int = 0,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.dilation = _single(dilation)

        self.groups = int(groups)
        self.eps = float(eps)
        self.chunk_N = int(chunk_N)

        if self.in_channels <= 0:
            raise ValueError(
                "in_channels must be positive, "
                f"got {self.in_channels}"
            )

        if self.out_channels <= 0:
            raise ValueError(
                "out_channels must be positive, "
                f"got {self.out_channels}"
            )

        if self.kernel_size <= 0:
            raise ValueError(
                "kernel_size must be positive, "
                f"got {self.kernel_size}"
            )

        if self.stride <= 0:
            raise ValueError(
                f"stride must be positive, got {self.stride}"
            )

        if self.padding < 0:
            raise ValueError(
                f"padding cannot be negative, got {self.padding}"
            )

        if self.dilation <= 0:
            raise ValueError(
                f"dilation must be positive, got {self.dilation}"
            )

        if self.groups <= 0:
            raise ValueError(
                f"groups must be positive, got {self.groups}"
            )

        if self.in_channels % self.groups != 0:
            raise ValueError(
                "in_channels must be divisible by groups"
            )

        if self.out_channels % self.groups != 0:
            raise ValueError(
                "out_channels must be divisible by groups"
            )

        self.cin_g = self.in_channels // self.groups
        self.cout_g = self.out_channels // self.groups

        self.K = self.cin_g * self.kernel_size

        self.N_factor = float(N_factor)

        if self.N_factor <= 0:
            raise ValueError(
                f"N_factor must be positive, got {self.N_factor}"
            )

        self.N = int(self.K * self.N_factor)

        if self.N <= 0:
            raise ValueError(
                "N_factor produced a non-positive number of projections: "
                f"K={self.K}, N_factor={self.N_factor}, N={self.N}"
            )

        self.weight = nn.Parameter(
            torch.randn(
                self.out_channels,
                self.cin_g,
                self.kernel_size,
            ) * 0.05
        )

        self.bias = (
            nn.Parameter(
                torch.zeros(self.out_channels)
            )
            if bias
            else None
        )


        self.G_seed = int(
            torch.randint(
                low=0,
                high=2**31 - 1,
                size=(1,),
            ).item()
        )
        G = make_G_conv1d_from_seed(
            seed=self.G_seed,
            groups=self.groups,
            N=self.N,
            cin_g=self.cin_g,
            kernel_size=self.kernel_size,
            device=torch.device("cpu"),
        )

        self.register_buffer(
            "G",
            G,
        )

    @property
    def effective_kernel_size(self) -> int:
        """
        Effective receptive-field length after dilation.
        """
        return (
            self.kernel_size - 1
        ) * self.dilation + 1


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "Conv1dGGD expects input with shape "
                "[batch_size, in_channels, length], "
                f"but received {tuple(x.shape)}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"but received {x.shape[1]}"
            )

        g = self.groups
        N = self.N
        cin_g = self.cin_g
        cout_g = self.cout_g
        kernel_size = self.kernel_size
        K = self.K

        x = x - x.mean(
            dim=(1, 2),
            keepdim=True,
        )


        Gx = self.G.to(
            device=x.device,
            dtype=x.dtype,
        )

        G_conv = Gx.reshape(
            g * N,
            cin_g,
            kernel_size,
        )

        z = F.conv1d(
            x,
            G_conv,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=g,
        )

        z_b = torch.where(
            z >= 0,
            torch.ones(
                (),
                device=z.device,
                dtype=z.dtype,
            ),
            -torch.ones(
                (),
                device=z.device,
                dtype=z.dtype,
            ),
        )

        W_flat = self.weight.reshape(
            g,
            cout_g,
            K,
        )

        Gw = self.G.to(
            device=x.device,
            dtype=W_flat.dtype,
        ).reshape(
            g,
            N,
            K,
        )
        W_proj = torch.bmm(
            Gw,
            W_flat.transpose(1, 2),
        )

        W_b = torch.where(
            W_proj >= 0,
            torch.ones(
                (),
                device=W_proj.device,
                dtype=W_proj.dtype,
            ),
            -torch.ones(
                (),
                device=W_proj.device,
                dtype=W_proj.dtype,
            ),
        )

        W_b = W_b.transpose(
            1,
            2,
        ).reshape(
            self.out_channels,
            N,
            1,
        )


        y_bin = F.conv1d(
            z_b,
            W_b.to(dtype=z_b.dtype),
            bias=None,
            stride=1,
            padding=0,
            dilation=1,
            groups=g,
        )

        y_bin = y_bin / float(N)


        x32 = x.float()
        W32 = self.weight.float()


        num = F.conv1d(
            x32,
            W32,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=g,
        )

        if self.dilation == 1:
            ones = torch.ones(
                g,
                cin_g,
                kernel_size,
                device=x.device,
                dtype=x32.dtype,
            )

            p2 = F.conv1d(
                x32.square(),
                ones,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=1,
                groups=g,
            )

        else:

            effective_kernel_size = self.effective_kernel_size

            ones = torch.zeros(
                g,
                cin_g,
                effective_kernel_size,
                device=x.device,
                dtype=x32.dtype,
            )

            ones[..., ::self.dilation] = 1.0

            p2 = F.conv1d(
                x32.square(),
                ones,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=1,
                groups=g,
            )

        inv_pn = torch.rsqrt(
            p2 + self.eps
        )

        inv_pn = inv_pn.repeat_interleave(
            cout_g,
            dim=1,
        )

        inv_wn = torch.rsqrt(
            W32.square().sum(
                dim=(1, 2)
            ) + self.eps
        ).reshape(
            1,
            self.out_channels,
            1,
        )

        corr = num * inv_pn * inv_wn


        corr = corr.clamp(
            min=-1.0 + 1e-4,
            max=1.0 - 1e-4,
        )

        y_surr = (
            2.0 / math.pi
        ) * torch.asin(corr)

        y_surr = y_surr.to(
            dtype=y_bin.dtype,
        )

        y = (
            y_bin.detach()
            + y_surr
            - y_surr.detach()
        )

        if self.bias is not None:
            y = y + self.bias.reshape(
                1,
                -1,
                1,
            )

        return y

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"N={self.N}, "
            f"N_factor={self.N_factor}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"dilation={self.dilation}, "
            f"groups={self.groups}, "
            f"bias={self.bias is not None}, "
            f"G_seed={self.G_seed}, "
            f"G_shape={tuple(self.G.shape)}"
        )
