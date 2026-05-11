import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.grad as tgrad

try:
    from torch.amp import custom_fwd, custom_bwd
except ImportError:
    def custom_fwd(**kwargs):
        def decorator(func):
            return func
        return decorator

    def custom_bwd(func=None, **kwargs):
        if func is None:
            def decorator(f):
                return f
            return decorator
        return func


_FWD_CHUNK_FLOOR = 1
_FWD_VRAM_FRACTION = 0.25
_FWD_CHUNK_FALLBACK = 4096


def _estimate_forward_chunk_1d(
    n: int,
    groups: int,
    cout_g: int,
    cin_g: int,
    k: int,
    batch_size: int,
    out_l: int,
    device: torch.device,
) -> int:
    """
    Estimate the largest forward-pass chunk that fits in VRAM.

    Per-chunk dominant allocations:
      1. G_conv weight slice: groups * chunk * Cin_g * k
      2. Z conv output: B * (groups * chunk) * outL
      3. W_proj matmul: groups * chunk * Cout_g
      4. 1x1 conv output accumulator: B * Cout * outL (fixed across loop)
    """
    if device.type != "cuda":
        return min(n, _FWD_CHUNK_FALLBACK)

    try:
        free, _total = torch.cuda.mem_get_info(device)
    except Exception:
        return min(n, _FWD_CHUNK_FALLBACK)

    budget = int(free * _FWD_VRAM_FRACTION)
    if torch.is_grad_enabled():
        budget //= 2

    bpf = 4
    cout = groups * cout_g

    accum_bytes = batch_size * cout * out_l * bpf

    g_conv_per_n = groups * cin_g * k * bpf
    z_per_n = batch_size * groups * out_l * bpf
    wproj_per_n = groups * cout_g * bpf

    cost_per_n = g_conv_per_n + z_per_n + wproj_per_n

    if cost_per_n <= 0:
        return min(n, _FWD_CHUNK_FALLBACK)

    effective_budget = max(0, budget - accum_bytes)
    chunk = max(_FWD_CHUNK_FLOOR, effective_budget // cost_per_n)
    return min(n, chunk)


def _make_dilated_ones_kernel_1d(
    groups: int,
    cin_g: int,
    k: int,
    dilation: int,
    ref: torch.Tensor,
) -> torch.Tensor:
    """
    Build a grouped kernel that computes the squared patch norm for a dilated conv.

    For dilation=1, shape is (groups, cin_g, k) filled with 1s.

    For dilation>1, the kernel has effective length:
        effective_k = (k - 1) * dilation + 1
    and contains ones at positions [0, dilation, 2*dilation, ...].
    """
    effective_k = (k - 1) * dilation + 1
    ones = ref.new_zeros(groups, cin_g, effective_k)
    ones[..., ::dilation] = 1
    return ones


class _Conv1d_GGD(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16, device_type="cuda")
    def forward(
        ctx,
        x,
        W,
        G,
        N: int,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
        eps: float,
        debug: bool,
        chunk_N: int,
    ):
        """
        x: (B, Cin, L)
        W: (Cout, Cin//groups, k)
        G: (groups, N, (Cin//groups)*k)

        Dilation is applied to both:
          - the G-based grouped conv
          - the W-based grouped conv
        """
        B, Cin, L = x.shape
        Cout, Cin_g, k = W.shape

        assert groups >= 1
        assert dilation >= 1
        assert Cin % groups == 0
        assert Cout % groups == 0
        assert Cin_g == (Cin // groups), "W has wrong Cin per group"

        K = Cin_g * k
        if G.dtype != x.dtype:
            G = G.to(x.dtype)

        assert G.shape == (groups, N, K), f"G must be (groups, N, {K})"
        assert int(N) == G.shape[1]

        ctx.stride = int(stride)
        ctx.padding = int(padding)
        ctx.dilation = int(dilation)
        ctx.groups = int(groups)
        ctx.eps = float(eps)
        ctx.debug = bool(debug)
        ctx.chunk_N = int(chunk_N) if chunk_N is not None else 0

        ctx.save_for_backward(x, W)

        Cout_g = Cout // groups

        W_flat = W.reshape(groups, Cout_g, K)
        W_flat_T = W_flat.transpose(1, 2).contiguous()

        effective_k = (k - 1) * dilation + 1
        out_l = (L + 2 * padding - effective_k) // stride + 1

        if ctx.chunk_N and 0 < ctx.chunk_N < N:
            effective_chunk = ctx.chunk_N
        else:
            effective_chunk = _estimate_forward_chunk_1d(
                N, groups, Cout_g, Cin_g, k, B, out_l, x.device
            )

        if effective_chunk < N:
            out = None
            for n0 in range(0, N, effective_chunk):
                n1 = min(N, n0 + effective_chunk)
                Nc = n1 - n0

                Gn = G[:, n0:n1, :]
                G_conv = Gn.reshape(groups * Nc, Cin_g, k)

                Z = F.conv1d(
                    x,
                    G_conv,
                    stride=ctx.stride,
                    padding=ctx.padding,
                    dilation=ctx.dilation,
                    groups=groups,
                )
                Z.sign_()

                W_proj = torch.bmm(Gn, W_flat_T)
                W_proj.sign_()

                W_b = W_proj.transpose(1, 2).reshape(Cout, Nc, 1)
                part = F.conv1d(Z, W_b, stride=1, padding=0, groups=groups)

                del Z, W_b, W_proj
                if out is None:
                    out = part
                else:
                    out += part
                del part

            Y = out.div_(float(N))
        else:
            G_conv = G.reshape(groups * N, Cin_g, k)
            Z = F.conv1d(
                x,
                G_conv,
                stride=ctx.stride,
                padding=ctx.padding,
                dilation=ctx.dilation,
                groups=groups,
            )
            Z.sign_()

            W_proj = torch.bmm(G, W_flat_T)
            W_proj.sign_()

            W_b = W_proj.transpose(1, 2).reshape(Cout, N, 1)
            Y = F.conv1d(Z, W_b, stride=1, padding=0, groups=groups)
            Y.div_(float(N))

            del Z, W_b, W_proj

        return Y

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, dy):
        x, W = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        eps = ctx.eps

        target_dtype = x.dtype
        if W.dtype != target_dtype:
            W = W.to(target_dtype)
        if dy.dtype != target_dtype:
            dy = dy.to(target_dtype)

        B, Cin, L = x.shape
        Cout, Cin_g, k = W.shape
        Cout_g = Cout // groups

        num = F.conv1d(
            x,
            W,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        ones = _make_dilated_ones_kernel_1d(
            groups=groups,
            cin_g=Cin_g,
            k=k,
            dilation=dilation,
            ref=x,
        )

        x_sq = x.square()
        p2 = F.conv1d(
            x_sq,
            ones,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=groups,
        )
        inv_pn = torch.rsqrt(p2 + eps)

        inv_pn_exp = inv_pn.repeat_interleave(Cout_g, dim=1)

        w2 = W.square().sum(dim=(1, 2))
        inv_wn = torch.rsqrt(w2 + eps)

        inv_denom = inv_pn_exp * inv_wn.view(1, Cout, 1)

        corr = (num * inv_denom).clamp_(-1.0 + 1e-4, 1.0 - 1e-4)

        d_asu_d_corr = (2.0 / torch.pi) * torch.rsqrt(1.0 - corr * corr)
        g = dy * d_asu_d_corr
        dL_dnum = g * inv_denom

        g_corr = g * corr
        tmp = g_corr.view(B, groups, Cout_g, *corr.shape[2:]).sum(dim=2)

        dL_dp2 = -0.5 * tmp * inv_pn.pow(2)
        del tmp

        dL_dx2 = tgrad.conv1d_input(
            x.shape,
            ones,
            dL_dp2,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=groups,
        )
        del dL_dp2, ones

        x_grad_norm = dL_dx2.mul_(2.0).mul_(x)

        x_grad_num = tgrad.conv1d_input(
            x.shape,
            W,
            dL_dnum,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        x_grad = x_grad_num.add_(x_grad_norm)
        del x_grad_num, x_grad_norm

        W_grad_num = tgrad.conv1d_weight(
            x,
            W.shape,
            dL_dnum,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        s = g_corr.sum(dim=(0, 2))
        del g_corr

        dL_dwn = -s * inv_wn
        W_grad_norm = W * (dL_dwn * inv_wn).view(Cout, 1, 1)

        W_grad = W_grad_num.add_(W_grad_norm)
        del W_grad_num, W_grad_norm

        return x_grad, W_grad, None, None, None, None, None, None, None, None, None


class Conv1dGGD(nn.Module):
    """
    Grouped 1D version of the binary-projection conv with arcsin surrogate backward.

    Notes:
      - groups must divide both in_channels and out_channels
      - G is stored per-group: (groups, N, (Cin/groups)*k)
      - forward uses grouped conv1d
      - backward uses analytic arcsin surrogate with group-aware patch norms
      - dilation is supported
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        N_factor: float = 1.0,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        eps: float = 1e-6,
        chunk_N: int = 0,
        debug: bool = False,
        device=None,
        dtype=None,
        std: float = 0.02
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.groups = int(groups)
        self.eps = float(eps)
        self.chunk_N = int(chunk_N) if chunk_N is not None else 0
        self.debug = bool(debug)

        assert self.groups >= 1
        assert self.dilation >= 1
        assert self.in_channels % self.groups == 0
        assert self.out_channels % self.groups == 0

        self.N_factor = float(N_factor)

        Cin_g = self.in_channels // self.groups
        k = self.kernel_size
        K = Cin_g * k

        self.N = max(500, int(K * self.N_factor))

        self.weight = nn.Parameter(
            torch.randn(self.out_channels, Cin_g, k, **factory_kwargs)
        )
        nn.init.normal_(self.weight, mean=0.0, std=std)

        self.bias = (
            nn.Parameter(torch.zeros(self.out_channels, **factory_kwargs))
            if bias else None
        )

        G = torch.randn(self.groups, self.N, K, **factory_kwargs)
        self.register_buffer("G", G)

        self.scale = nn.Parameter(torch.ones(self.out_channels, **factory_kwargs))

        self._G_gpu = None

    @torch.no_grad()
    def resample_G(self):
        self.G.copy_(torch.randn_like(self.G))
        self._G_gpu = None

    @property
    def effective_kernel_size(self) -> int:
        return (self.kernel_size - 1) * self.dilation + 1

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"N_factor={self.N_factor}"
        )

        if self.padding != 0:
            s += f", padding={self.padding}"
        if self.dilation != 1:
            s += f", dilation={self.dilation}"
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
        G = self._G_gpu
        if G is None or G.device != x.device:
            if self.G.device == x.device:
                G = self.G
            else:
                G = self.G.to(x.device, non_blocking=True)

            try:
                self._G_gpu = G
            except Exception:
                pass

        output = _Conv1d_GGD.apply(
            x,
            self.weight,
            G,
            self.N,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.eps,
            self.debug,
            self.chunk_N,
        )

        output = output * self.scale.view(1, -1, 1)

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)

        return output