import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# IRNet binary quantizer
# -----------------------------------------------------------------------------

class BinaryQuantize(torch.autograd.Function):
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


def _safe_std(x: torch.Tensor, dim, keepdim=True, eps: float = 1e-5):
    return x.std(dim=dim, keepdim=keepdim, unbiased=False).clamp_min(eps)


def _irnet_pow2_scale(mean_abs: torch.Tensor) -> torch.Tensor:
    """
    sw = 2^( round(log2(mean_abs)) )
    with a clamp for numerical safety.
    """
    mean_abs = mean_abs.clamp_min(1e-8)
    return torch.pow(
        torch.tensor(2.0, device=mean_abs.device, dtype=mean_abs.dtype),
        torch.round(torch.log(mean_abs) / math.log(2.0))
    )


# -----------------------------------------------------------------------------
# Linear IRNet
# -----------------------------------------------------------------------------

class LinearIRNet(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        binary_input: bool = True,
        binary_weight: bool = True,
        k: float = 10.0,
        t: float = 0.1,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.binary_input = binary_input
        self.binary_weight = binary_weight

        self.register_buffer("k", torch.tensor([k], dtype=torch.float32))
        self.register_buffer("t", torch.tensor([t], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight

        if self.binary_weight:
            # Normalize per output neuron
            bw = w - w.mean(dim=1, keepdim=True)
            bw = bw / _safe_std(bw, dim=1, keepdim=True)

            # sw shape: [out_features, 1]
            mean_abs = bw.abs().mean(dim=1, keepdim=True)
            sw = _irnet_pow2_scale(mean_abs).detach()

            bw = BinaryQuantize.apply(bw, self.k.to(w.device), self.t.to(w.device))
            bw = bw * sw
        else:
            bw = w

        if self.binary_input:
            ba = BinaryQuantize.apply(x, self.k.to(x.device), self.t.to(x.device))
        else:
            ba = x

        return F.linear(ba, bw, self.bias)


# -----------------------------------------------------------------------------
# Conv1d IRNet
# -----------------------------------------------------------------------------

class Conv1dIRNet(nn.Conv1d):
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
        binary_weight: bool = True,
        k: float = 10.0,
        t: float = 0.1,
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
        self.binary_weight = binary_weight

        self.register_buffer("k", torch.tensor([k], dtype=torch.float32))
        self.register_buffer("t", torch.tensor([t], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight

        if self.binary_weight:
            # Normalize per output channel over [Cin/groups, K]
            bw = w - w.mean(dim=(1, 2), keepdim=True)
            bw = bw / _safe_std(bw, dim=(1, 2), keepdim=True)

            # sw shape: [out_channels, 1, 1]
            mean_abs = bw.abs().mean(dim=(1, 2), keepdim=True)
            sw = _irnet_pow2_scale(mean_abs).detach()

            bw = BinaryQuantize.apply(bw, self.k.to(w.device), self.t.to(w.device))
            bw = bw * sw
        else:
            bw = w

        if self.binary_input:
            ba = BinaryQuantize.apply(x, self.k.to(x.device), self.t.to(x.device))
        else:
            ba = x

        return F.conv1d(
            ba,
            bw,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )