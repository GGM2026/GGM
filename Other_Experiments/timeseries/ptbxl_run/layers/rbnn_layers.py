import math
import numpy as np
from scipy.stats import ortho_group

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def get_ab(N: int):
    """
    Factor N into a * b, choosing a and b as close as possible.
    """
    sqrt = int(np.sqrt(N))
    for i in range(sqrt, 0, -1):
        if N % i == 0:
            return i, N // i
    return 1, N


def _safe_std(x: torch.Tensor, dim, keepdim=True, eps: float = 1e-5):
    return x.std(dim=dim, keepdim=keepdim, unbiased=False).clamp_min(eps)


def _make_orthogonal(n: int, dtype=torch.float32):
    return torch.tensor(ortho_group.rvs(dim=n), dtype=dtype)


# -----------------------------------------------------------------------------
# Binary quantizers from RBNN-style code
# -----------------------------------------------------------------------------

class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * (2 * torch.sqrt(t**2 / 2) - torch.abs(t**2 * input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input, None, None


class BinaryQuantizeAct(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        k = torch.tensor(1.0, device=input.device, dtype=input.dtype)
        t = torch.maximum(
            t.to(device=input.device, dtype=input.dtype),
            torch.tensor(1.0, device=input.device, dtype=input.dtype),
        )
        grad_input = k * (2 * torch.sqrt(t**2 / 2) - torch.abs(t**2 * input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input, None, None


# -----------------------------------------------------------------------------
# Shared RBNN rotation mixin
# -----------------------------------------------------------------------------

class _RBNNMixin:
    """
    Shared rotation logic for Conv1d / Conv2d / Linear.

    Expected from subclass:
      - self.weight
      - self.a, self.b
      - self.R1, self.R2
      - self.rotate
      - self.epoch
      - self.rotation_update
    """

    def _maybe_update_rotation(self, X: torch.Tensor):
        """
        X shape: [out_channels_or_features, a, b]
        """
        if self.epoch <= -1:
            return

        if self.rotation_update <= 0:
            return

        if self.epoch % self.rotation_update != 0:
            return

        with torch.no_grad():
            Xd = X.detach()

            for _ in range(3):
                # V = R1^T X R2
                V = torch.matmul(
                    torch.matmul(self.R1.t().unsqueeze(0), Xd),
                    self.R2.unsqueeze(0),
                )
                B = torch.sign(V)

                # D1 = sum_i B_i R2^T X_i^T
                D1 = torch.zeros_like(self.R1)
                for Bi, Xi in zip(B, Xd):
                    D1 += Bi @ self.R2.t() @ Xi.t()
                U1, _, Vh1 = torch.linalg.svd(D1, full_matrices=False)
                self.R1.copy_(Vh1.transpose(-2, -1) @ U1.transpose(-2, -1))

                # D2 = sum_i X_i^T R1 B_i
                D2 = torch.zeros_like(self.R2)
                for Xi, Bi in zip(Xd, B):
                    D2 += Xi.t() @ self.R1 @ Bi
                U2, _, Vh2 = torch.linalg.svd(D2, full_matrices=False)
                self.R2.copy_(U2 @ Vh2)

    def _rotated_weight(self, w_norm: torch.Tensor):
        """
        w_norm shape:
          Conv1d: [out_channels, in_channels/groups, kernel_size]
          Conv2d: [out_channels, in_channels/groups, kh, kw]
          Linear: [out_features, in_features]
        """
        X = w_norm.view(w_norm.shape[0], self.a, self.b)
        self._maybe_update_rotation(X)

        Rweight = torch.matmul(
            torch.matmul(self.R1.t().unsqueeze(0), X),
            self.R2.unsqueeze(0),
        ).view_as(w_norm)

        delta = Rweight.detach() - w_norm
        w_mix = w_norm + torch.abs(torch.sin(self.rotate)) * delta
        return w_mix


# -----------------------------------------------------------------------------
# Conv1d RBNN
# -----------------------------------------------------------------------------

class Conv1dRBNN(_RBNNMixin, nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        binary_input: bool = True,
        k: float = 10.0,
        t: float = 0.1,
        rotation_update: int = 1,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.binary_input = binary_input
        self.rotation_update = rotation_update
        self.epoch = -1

        self.register_buffer("k", torch.tensor([k], dtype=torch.float32))
        self.register_buffer("t", torch.tensor([t], dtype=torch.float32))

        # weight shape: [out_channels, in_channels/groups, kernel_size]
        N = int(np.prod(self.weight.shape[1:]))
        self.a, self.b = get_ab(N)

        self.register_buffer("R1", _make_orthogonal(self.a))
        self.register_buffer("R2", _make_orthogonal(self.b))

        # per-output-channel scaling, stored as [out_channels, 1]
        sw = (
            self.weight.detach()
            .abs()
            .view(self.weight.size(0), -1)
            .mean(dim=-1, keepdim=True)
        )
        self.alpha = nn.Parameter(sw, requires_grad=True)

        # broadcastable to [out_channels, in_channels/groups, kernel_size]
        self.rotate = nn.Parameter(
            torch.ones(self.weight.size(0), 1, 1) * (math.pi / 2),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # activation normalization over [C, L]
        a1 = x - x.mean(dim=(1, 2), keepdim=True)
        a2 = a1 / _safe_std(a1, dim=(1, 2), keepdim=True)

        # weight normalization per output channel
        w1 = self.weight - self.weight.mean(dim=(1, 2), keepdim=True)
        w2 = w1 / _safe_std(w1, dim=(1, 2), keepdim=True)

        # rotation-guided interpolation
        w3 = self._rotated_weight(w2)

        # binarize
        bw = BinaryQuantize.apply(w3, self.k.to(w3.device), self.t.to(w3.device))
        if self.binary_input:
            ba = BinaryQuantizeAct.apply(a2, self.k.to(a2.device), self.t.to(a2.device))
        else:
            ba = a2

        out = F.conv1d(
            ba,
            bw,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        # [B, Cout, L] * [1, Cout, 1]
        out = out * self.alpha.view(1, -1, 1)
        return out



# -----------------------------------------------------------------------------
# Linear RBNN
# -----------------------------------------------------------------------------

class LinearRBNN(_RBNNMixin, nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        binary_input: bool = True,
        k: float = 10.0,
        t: float = 0.1,
        rotation_update: int = 1,
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)

        self.binary_input = binary_input
        self.rotation_update = rotation_update
        self.epoch = -1

        self.register_buffer("k", torch.tensor([k], dtype=torch.float32))
        self.register_buffer("t", torch.tensor([t], dtype=torch.float32))

        # weight shape: [out_features, in_features]
        N = int(self.weight.shape[1])
        self.a, self.b = get_ab(N)

        self.register_buffer("R1", _make_orthogonal(self.a))
        self.register_buffer("R2", _make_orthogonal(self.b))

        # per-output-neuron scaling, stored as [out_features, 1]
        sw = self.weight.detach().abs().mean(dim=1, keepdim=True)
        self.alpha = nn.Parameter(sw, requires_grad=True)

        # broadcastable to [out_features, in_features]
        self.rotate = nn.Parameter(
            torch.ones(self.weight.size(0), 1) * (math.pi / 2),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # activation normalization over feature dimension
        a1 = x - x.mean(dim=-1, keepdim=True)
        a2 = a1 / _safe_std(a1, dim=-1, keepdim=True)

        # weight normalization per output neuron
        w1 = self.weight - self.weight.mean(dim=1, keepdim=True)
        w2 = w1 / _safe_std(w1, dim=1, keepdim=True)

        # rotation-guided interpolation
        w3 = self._rotated_weight(w2)

        # binarize
        bw = BinaryQuantize.apply(w3, self.k.to(w3.device), self.t.to(w3.device))
        if self.binary_input:
            ba = BinaryQuantizeAct.apply(a2, self.k.to(a2.device), self.t.to(a2.device))
        else:
            ba = a2

        out = F.linear(ba, bw, self.bias)

        # broadcast alpha to [..., out_features]
        alpha = self.alpha.squeeze(-1)
        view_shape = [1] * (out.dim() - 1) + [alpha.numel()]
        out = out * alpha.view(*view_shape)
        return out


# -----------------------------------------------------------------------------
# Optional helper
# -----------------------------------------------------------------------------

def set_rbnn_epoch(module: nn.Module, epoch: int):
    """
    Call once per epoch so rotation matrices can refresh.
    """
    for m in module.modules():
        if isinstance(m, (Conv1dRBNN, Conv2dRBNN, LinearRBNN)):
            m.epoch = epoch