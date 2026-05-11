import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.grad as tgrad

try:
    from torch.cuda.amp import custom_fwd, custom_bwd
except ImportError:
    def custom_fwd(**kwargs):
        def decorator(func): return func
        return decorator
    def custom_bwd(func): return func
class _Conv2d_GGD(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)     
    def forward(
        ctx,
        x, W, G,
        N: int, stride: int, padding: int, groups: int,
        eps: float, debug: bool, chunk_N: int
    ):
        """
        x: (B, Cin, H, W)
        W: (Cout, Cin//groups, kH, kW)
        G: (groups, N, (Cin//groups)*kH*kW)
        """
        B, Cin, H, Win = x.shape
        Cout, Cin_g, kH, kW = W.shape

        assert groups >= 1
        assert Cin % groups == 0
        assert Cout % groups == 0
        assert Cin_g == (Cin // groups), "W has wrong Cin per group"
        K = Cin_g * kH * kW
        if G.dtype != x.dtype:
            G = G.to(x.dtype)

        assert G.shape == (groups, N, K), f"G must be (groups,N,{K})"
        assert int(N) == G.shape[1]

        ctx.stride = int(stride)
        ctx.padding = int(padding)
        ctx.groups = int(groups)
        ctx.eps = float(eps)
        ctx.debug = bool(debug)
        ctx.chunk_N = int(chunk_N) if chunk_N is not None else 0

        ctx.save_for_backward(x, W)

        Cout_g = Cout // groups
        W_flat = W.view(groups, Cout_g, K)

        if ctx.chunk_N and 0 < ctx.chunk_N < N:
            out = None
            for n0 in range(0, N, ctx.chunk_N):
                n1 = min(N, n0 + ctx.chunk_N)
                Nc = n1 - n0

                Gn = G[:, n0:n1, :]
                G_conv = Gn.contiguous().view(groups * Nc, Cin_g, kH, kW)

                Z = F.conv2d(x, G_conv, stride=ctx.stride, padding=ctx.padding, groups=groups)
                Z.sign_()

                W_proj = torch.bmm(Gn, W_flat.transpose(1, 2))
                W_proj.sign_()

                W_b = W_proj.transpose(1, 2).contiguous().view(Cout, Nc, 1, 1)

                part = F.conv2d(Z, W_b, stride=1, padding=0, groups=groups)
                out = part if out is None else (out + part)

            Y = out / float(N)

        else:
            G_conv = G.contiguous().view(groups * N, Cin_g, kH, kW)
            Z = F.conv2d(x, G_conv, stride=ctx.stride, padding=ctx.padding, groups=groups)
            Z.sign_()

            W_proj = torch.bmm(G, W_flat.transpose(1, 2))
            W_proj.sign_()

            W_b = W_proj.transpose(1, 2).contiguous().view(Cout, N, 1, 1)
            Y = F.conv2d(Z, W_b, stride=1, padding=0, groups=groups) / float(N)

        return Y

    @staticmethod
    @custom_bwd
    def backward(ctx, dy):
        x, W = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        eps = ctx.eps
        target_dtype = x.dtype
        if W.dtype != target_dtype:
            W = W.to(target_dtype)
        if dy.dtype != target_dtype:
            dy = dy.to(target_dtype)

        B, Cin, H, Win = x.shape
        Cout, Cin_g, kH, kW = W.shape
        Cout_g = Cout // groups

        num = F.conv2d(x, W, stride=stride, padding=padding, groups=groups)

        ones = x.new_ones(groups, Cin_g, kH, kW)
        p2 = F.conv2d(x * x, ones, stride=stride, padding=padding, groups=groups)
        pn = torch.sqrt(p2 + eps)

        pn_exp = pn.repeat_interleave(Cout_g, dim=1)

        w2 = (W * W).sum(dim=(1, 2, 3))
        wn = torch.sqrt(w2 + eps)

        denom = pn_exp * wn.view(1, Cout, 1, 1)
        corr = (num / denom).clamp(-1.0 + 1e-4, 1.0 - 1e-4)

        d_asu_d_corr = (2.0 / torch.pi) * torch.rsqrt(1.0 - corr * corr)
        g = dy * d_asu_d_corr
        dL_dnum = g / denom

        tmp = (g * corr).view(B, groups, Cout_g, *corr.shape[2:]).sum(dim=2)
        dL_dpn = -tmp / (pn + eps)
        dL_dp2 = 0.5 * dL_dpn / (pn + eps)

        dL_dx2 = tgrad.conv2d_input(x.shape, ones, dL_dp2, stride=stride, padding=padding, groups=groups,)
        x_grad_norm = 2.0 * x * dL_dx2

        x_grad_num = tgrad.conv2d_input(x.shape, W, dL_dnum, stride=stride, padding=padding, groups=groups)
        x_grad = x_grad_num + x_grad_norm

        W_grad_num = tgrad.conv2d_weight(x, W.shape, dL_dnum, stride=stride, padding=padding, groups=groups)

        s = (g * corr).sum(dim=(0, 2, 3))
        dL_dwn = -s / (wn + eps)
        W_grad_norm = W * (dL_dwn / (wn + eps)).view(Cout, 1, 1, 1)

        W_grad = W_grad_num + W_grad_norm

        return x_grad, W_grad, None, None, None, None, None, None, None, None


class Conv2dGGD(nn.Module):
    """
    Grouped version of the binary-projection conv with arcsin surrogate backward.

    Notes:
      - groups must divide both in_channels and out_channels
      - G is stored per-group: (groups, N, (Cin/groups)*k*k)
      - forward uses grouped convs
      - backward uses analytic arcsin surrogate with group-aware patch norms
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        N_scale: float = 1.0,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
        eps: float = 1e-6,
        chunk_N: int = 0,
        debug: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.groups = int(groups)
        self.eps = float(eps)
        self.chunk_N = int(chunk_N) if chunk_N is not None else 0
        self.debug = bool(debug)

        assert self.groups >= 1
        assert self.in_channels % self.groups == 0
        assert self.out_channels % self.groups == 0

        self.N_scale = float(N_scale)
        Cin_g = self.in_channels // self.groups
        k = self.kernel_size
        K = Cin_g * k * k
        self.N = int(Cin_g * k * k * self.N_scale)
        
        self.weight = nn.Parameter(torch.randn(self.out_channels, Cin_g, k, k, **factory_kwargs))
        self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None
        G = torch.randn(self.groups, self.N, K, **factory_kwargs)
        self.register_buffer("G", G)

        self.scale = nn.Parameter(torch.randn(self.out_channels))

    @torch.no_grad()
    def resample_G(self):
        self.G.copy_(torch.randn_like(self.G))

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size=({self.kernel_size}, {self.kernel_size}), "
            f"stride=({self.stride}, {self.stride}), "
            f"N_scale: {self.N_scale}"
        )

        if self.padding != 0:
            s += f", padding=({self.padding}, {self.padding})"
        if self.groups != 1:
            s += f", groups={self.groups}"

        if self.N is not None:
            s += f", N={self.N}"
        if self.chunk_N not in (0, None):
            s += f", chunk_N={self.chunk_N}"
        if self.eps != 1e-6:
            s += f", eps={self.eps:g}"
        if self.debug:
            s += f", debug={self.debug}"

        s += f", G_shape={tuple(self.G.shape)}"
        return s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output =  _Conv2d_GGD.apply(
            x, self.weight, self.G,
            self.N, self.stride, self.padding, self.groups,
            self.eps, self.debug, self.chunk_N
        )

        output = output * self.scale.view(1, -1, 1, 1)

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        return output
