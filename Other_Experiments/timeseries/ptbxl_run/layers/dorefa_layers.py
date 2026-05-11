import torch
import torch.nn as nn
import torch.nn.functional as F



class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SignSTE(torch.autograd.Function):
    """
    BWN sign function with STE, matching the official:
        tf.where(tf.equal(x, 0), tf.ones_like(x), tf.sign(x / E)) * E
    Zero maps to +E (positive one before scaling).
    """
    @staticmethod
    def forward(ctx, x):
        return torch.where(x == 0, torch.ones_like(x), torch.sign(x))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return RoundSTE.apply(x)


def sign_ste(x: torch.Tensor) -> torch.Tensor:
    return SignSTE.apply(x)



def quantize_kbit(x: torch.Tensor, k: int) -> torch.Tensor:
    """Uniform k-bit quantization on [0, 1]."""
    n = float(2 ** k - 1)
    return round_ste(x * n) / n


def dorefa_quantize_weight(w: torch.Tensor, k: int) -> torch.Tensor:
    """
    Faithful port of fw() from dorefa.py.

    k=1  → Binary Weight Network (BWN):
               sign(w) * E,  E = mean(|w|),  full STE on sign op
    k≥2  → k-bit:
               2 * quantize_k( tanh(w) / (2 * max|tanh(w)|) + 0.5 ) - 1
    k=32 → full precision passthrough
    """
    if k >= 32:
        return w

    if k == 1:
        E = w.abs().mean().detach()
        return sign_ste(w) * E

    w_tanh = torch.tanh(w)
    max_abs = w_tanh.abs().max().detach()
    w_norm = w_tanh / (max_abs + 1e-8) * 0.5 + 0.5
    return 2.0 * quantize_kbit(w_norm, k) - 1.0


def dorefa_quantize_activation(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Faithful port of fa() from dorefa.py.

    The official fa() is simply quantize(x, k) with NO built-in clip.
    The clip to [0, 1] is the caller's responsibility (nonlin → fa pattern).

    If you are plugging this into a plain ReLU network you should apply
    torch.clamp(x, 0.0, 1.0) *before* calling this function, exactly as
    the reference model does with its separate nonlin() + activate() calls.
    """
    if k >= 32:
        return x
    return quantize_kbit(x, k)



class LinearDoReFa(nn.Linear):
    """
    Fully-connected layer with DoReFa-Net weight + activation quantization.

    Parameters
    ----------
    quantize_input : bool
        If True the input is treated as an activation and quantized with fa().
        The caller is responsible for clipping to [0, 1] befoauthord
        (or set quantize_input=False for the first / last layer).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_bits: int = 1,
        act_bits: int = 4,
        quantize_input: bool = True,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.quantize_input = quantize_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = dorefa_quantize_activation(x, self.act_bits) if self.quantize_input else x
        w_q = dorefa_quantize_weight(self.weight, self.weight_bits)
        return F.linear(x_q, w_q, self.bias)


class Conv1dDoReFa(nn.Conv1d):
    """Conv1d counterpart of LinearDoReFa — same semantics."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias: bool = True,
        weight_bits: int = 1,
        act_bits: int = 4,
        quantize_input: bool = True,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias,
        )
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.quantize_input = quantize_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = dorefa_quantize_activation(x, self.act_bits) if self.quantize_input else x
        w_q = dorefa_quantize_weight(self.weight, self.weight_bits)
        return F.conv1d(
            x_q, w_q, self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups,
        )