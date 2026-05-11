import torch
import torch.nn as nn
import torch.nn.functional as F


class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x.gt(1.0)] = 0
        grad_input[x.lt(-1.0)] = 0
        return grad_input


def ste_sign(x: torch.Tensor) -> torch.Tensor:
    return SignSTE.apply(x)


def _safe_sign(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def binarize_activation(x: torch.Tensor) -> torch.Tensor:
    x_bin = ste_sign(x)
    x_bin = torch.where(x_bin == 0, _safe_sign(x), x_bin)
    return x_bin


def xnor_input_scale_linear(x: torch.Tensor) -> torch.Tensor:
    """
    Input scaling for linear layers.
    x shape can be [..., in_features]
    Returns scale with shape [..., 1]
    """
    return x.abs().mean(dim=-1, keepdim=True).detach()


def xnor_input_scale_conv1d(x: torch.Tensor) -> torch.Tensor:
    """
    Input scaling for Conv1d.
    x shape: [B, C, L]
    Returns scale with shape [B, 1, 1]
    """
    return x.abs().mean(dim=(1, 2), keepdim=True).detach()


def xnor_binarize_input_linear(x: torch.Tensor) -> torch.Tensor:
    alpha_x = xnor_input_scale_linear(x)
    x_bin = binarize_activation(x)
    return alpha_x * x_bin


def xnor_binarize_input_conv1d(x: torch.Tensor) -> torch.Tensor:
    alpha_x = xnor_input_scale_conv1d(x)
    x_bin = binarize_activation(x)
    return alpha_x * x_bin


def xnor_binarize_weight_linear(w: torch.Tensor) -> torch.Tensor:
    """
    w shape: [out_features, in_features]
    Per-output-channel scaling.
    """
    alpha_w = w.abs().mean(dim=1, keepdim=True).detach()
    w_bin = ste_sign(w)
    w_bin = torch.where(w_bin == 0, _safe_sign(w), w_bin)
    return alpha_w * w_bin


def xnor_binarize_weight_conv1d(w: torch.Tensor) -> torch.Tensor:
    """
    w shape: [out_channels, in_channels/groups, kernel_size]
    Per-output-channel scaling.
    """
    alpha_w = w.abs().mean(dim=(1, 2), keepdim=True).detach()
    w_bin = ste_sign(w)
    w_bin = torch.where(w_bin == 0, _safe_sign(w), w_bin)
    return alpha_w * w_bin


class LinearXNORNet(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        binary_input: bool = True,
        binary_weight: bool = True,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.binary_input = binary_input
        self.binary_weight = binary_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.binary_input:
            x_q = xnor_binarize_input_linear(x)
        else:
            x_q = x

        if self.binary_weight:
            w_q = xnor_binarize_weight_linear(self.weight)
        else:
            w_q = self.weight

        return F.linear(x_q, w_q, self.bias)


class Conv1dXNORNet(nn.Conv1d):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.binary_input:
            x_q = xnor_binarize_input_conv1d(x)
        else:
            x_q = x

        if self.binary_weight:
            w_q = xnor_binarize_weight_conv1d(self.weight)
        else:
            w_q = self.weight

        return F.conv1d(
            x_q,
            w_q,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )