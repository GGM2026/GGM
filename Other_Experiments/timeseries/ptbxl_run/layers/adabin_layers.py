import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        k = k.to(input.device, input.dtype)
        t = t.to(input.device, input.dtype)
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None


class BinaryActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_a = nn.Parameter(torch.tensor(1.0))
        self.beta_a = nn.Parameter(torch.tensor(0.0))

    def gradient_approx(self, x):
        out_forward = torch.sign(x)

        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1

        out1 = (-1) * mask1.to(x.dtype) + (x * x + 2 * x) * (1 - mask1.to(x.dtype))
        out2 = out1 * mask2.to(x.dtype) + (-x * x + 2 * x) * (1 - mask2.to(x.dtype))
        out3 = out2 * mask3.to(x.dtype) + 1 * (1 - mask3.to(x.dtype))

        return out_forward.detach() - out3.detach() + out3

    def forward(self, x):
        x = (x - self.beta_a) / self.alpha_a
        x = self.gradient_approx(x)
        return self.alpha_a * (x + self.beta_a)


class Conv1dAdaBin(nn.Conv1d):
    """
    1-bit AdaBin Conv1d.
    Always uses binary activations and binary weights.
    Weight shape: [out_channels, in_channels/groups, kernel_size]
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
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

        self.register_buffer("k", torch.tensor([10.0], dtype=torch.float32))
        self.register_buffer("t", torch.tensor([0.1], dtype=torch.float32))
        self.binary_a = BinaryActivation()

        if isinstance(self.kernel_size, tuple):
            ksize = self.kernel_size[0]
        else:
            ksize = self.kernel_size

        self.filter_size = ksize * (in_channels // groups)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # 1-bit activation
        inputs = self.binary_a(inputs)

        # 1-bit weight
        w = self.weight
        beta_w = w.mean(dim=(1, 2), keepdim=True)  # [out_channels, 1, 1]
        alpha_w = torch.sqrt(
            ((w - beta_w) ** 2).sum(dim=(1, 2), keepdim=True) / self.filter_size
        ).clamp_min(1e-8)

        w_norm = (w - beta_w) / alpha_w
        wb = BinaryQuantize.apply(w_norm, self.k, self.t)
        weight = wb * alpha_w + beta_w

        return F.conv1d(
            inputs,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class LinearAdaBin(nn.Linear):
    """
    1-bit AdaBin Linear.
    Always uses binary activations and binary weights.
    Weight shape: [out_features, in_features]
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)

        self.register_buffer("k", torch.tensor([10.0], dtype=torch.float32))
        self.register_buffer("t", torch.tensor([0.1], dtype=torch.float32))
        self.binary_a = BinaryActivation()
        self.filter_size = in_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # 1-bit activation
        inputs = self.binary_a(inputs)

        # 1-bit weight
        w = self.weight
        beta_w = w.mean(dim=1, keepdim=True)  # [out_features, 1]
        alpha_w = torch.sqrt(
            ((w - beta_w) ** 2).sum(dim=1, keepdim=True) / self.filter_size
        ).clamp_min(1e-8)

        w_norm = (w - beta_w) / alpha_w
        wb = BinaryQuantize.apply(w_norm, self.k, self.t)
        weight = wb * alpha_w + beta_w

        return F.linear(inputs, weight, self.bias)

class Maxout(nn.Module):
    def __init__(self, channel, neg_init=0.25, pos_init=1.0):
        super().__init__()
        self.neg_scale = nn.Parameter(neg_init * torch.ones(channel))
        self.pos_scale = nn.Parameter(pos_init * torch.ones(channel))
        self.relu = nn.ReLU()

    def forward(self, x):
        return (
            self.pos_scale.view(1, -1, 1) * self.relu(x)
            - self.neg_scale.view(1, -1, 1) * self.relu(-x)
        )