import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.grad as tgrad

try:
    from torch.amp import custom_fwd, custom_bwd
except ImportError:
    # Fallback for older PyTorch versions
    def custom_fwd(**kwargs):
        def decorator(func): return func
        return decorator
    def custom_bwd(func): return func


# ---------------------------------------------------------------------------
# Dynamic VRAM-aware chunk sizing (ported from GGM/layers.py)
# ---------------------------------------------------------------------------
_FWD_CHUNK_FLOOR = 1       # never chunk smaller than this
_FWD_VRAM_FRACTION = 0.25  # conservative: model + input + accumulator also live in VRAM
_FWD_CHUNK_FALLBACK = 4096 # fallback when CUDA info unavailable


def _estimate_forward_chunk(
    n: int,
    groups: int,
    cout_g: int,
    cin_g: int,
    kH: int,
    kW: int,
    batch_size: int,
    out_h: int,
    out_w: int,
    device: torch.device,
) -> int:
    """Estimate the largest forward-pass chunk that fits in VRAM.

    Per-chunk dominant allocations:
      1. G_conv weight slice: chunk × Cin_g × kH × kW (per group, but groups
         are concatenated along dim-0 → groups × chunk total rows)
      2. Z conv output: B × (groups × chunk) × outH × outW
      3. W_proj matmul: groups × chunk × Cout_g
      4. 1×1 conv output: B × Cout × outH × outW

    We size chunk so these fit within a fraction of free VRAM.
    """
    if device.type != "cuda":
        return min(n, _FWD_CHUNK_FALLBACK)
    try:
        free, _total = torch.cuda.mem_get_info(device)
    except Exception:
        return min(n, _FWD_CHUNK_FALLBACK)

    budget = int(free * _FWD_VRAM_FRACTION)
    if torch.is_grad_enabled():
        budget //= 2  # autograd graph metadata overhead

    bpf = 4  # bytes per float32
    cout = groups * cout_g

    # Fixed accumulator cost (lives for entire loop)
    accum_bytes = batch_size * cout * out_h * out_w * bpf

    # Per-chunk-N costs:
    g_conv_per_n = groups * cin_g * kH * kW * bpf
    z_per_n = batch_size * groups * out_h * out_w * bpf
    wproj_per_n = groups * cout_g * bpf
    # The 1x1 conv output is constant size (B,Cout,outH,outW) — reused
    cost_per_n = g_conv_per_n + z_per_n + wproj_per_n

    if cost_per_n <= 0:
        return min(n, _FWD_CHUNK_FALLBACK)

    effective_budget = max(0, budget - accum_bytes)
    chunk = max(_FWD_CHUNK_FLOOR, effective_budget // cost_per_n)
    return min(n, chunk)


class _Conv2d_GGD(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16, device_type='cuda')     
    def forward(
        ctx,
        x, W, G,                 # tensors
        N: int, stride: int, padding: int, groups: int,  # ints
        eps: float, debug: bool, chunk_N: int            # misc
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
        # Pre-compute W_flat and its transpose once (reused across chunks)
        W_flat = W.reshape(groups, Cout_g, K)
        W_flat_T = W_flat.transpose(1, 2).contiguous()  # (g, K, Cout_g)

        # ---- Compute output spatial dims for dynamic chunk sizing ----
        out_h = (H + 2 * padding - kH) // stride + 1
        out_w = (Win + 2 * padding - kW) // stride + 1

        # ---- Determine chunk size: use dynamic VRAM sizing if no explicit chunk_N ----
        if ctx.chunk_N and 0 < ctx.chunk_N < N:
            effective_chunk = ctx.chunk_N
        else:
            effective_chunk = _estimate_forward_chunk(
                N, groups, Cout_g, Cin_g, kH, kW,
                B, out_h, out_w, x.device,
            )

        # ---- forward (binary) ----
        if effective_chunk < N:
            out = None
            for n0 in range(0, N, effective_chunk):
                n1 = min(N, n0 + effective_chunk)
                Nc = n1 - n0

                Gn = G[:, n0:n1, :]  # (g, Nc, K) — already contiguous slice
                G_conv = Gn.reshape(groups * Nc, Cin_g, kH, kW)

                Z = F.conv2d(x, G_conv, stride=ctx.stride, padding=ctx.padding, groups=groups)
                Z.sign_()  # in-place

                # W_proj: (g, Nc, Cout_g) = (g, Nc, K) @ (g, K, Cout_g)
                W_proj = torch.bmm(Gn, W_flat_T)
                W_proj.sign_()  # in-place

                # 1x1 grouped conv weights: (Cout, Nc, 1, 1)
                W_b = W_proj.transpose(1, 2).reshape(Cout, Nc, 1, 1)

                part = F.conv2d(Z, W_b, stride=1, padding=0, groups=groups)
                del Z, W_b, W_proj
                if out is None:
                    out = part
                else:
                    out += part  # accumulate in-place
                del part

            Y = out.div_(float(N))  # in-place division

        else:
            G_conv = G.reshape(groups * N, Cin_g, kH, kW)
            Z = F.conv2d(x, G_conv, stride=ctx.stride, padding=ctx.padding, groups=groups)
            Z.sign_()

            W_proj = torch.bmm(G, W_flat_T)
            W_proj.sign_()

            W_b = W_proj.transpose(1, 2).reshape(Cout, N, 1, 1)
            Y = F.conv2d(Z, W_b, stride=1, padding=0, groups=groups)
            Y.div_(float(N))  # in-place
            del Z, W_b, W_proj

        return Y

    @staticmethod
    @custom_bwd(device_type='cuda')
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

        # ---- arcsin surrogate backward (analytic), group-aware ----
        # num: (B, Cout, outH, outW)
        num = F.conv2d(x, W, stride=stride, padding=padding, groups=groups)

        # patch norm per group: p2, pn: (B, groups, outH, outW)
        # Reuse a single ones-buffer (register_buffer would be better in
        # the module, but autograd.Function has no persistent state).
        ones = x.new_ones(groups, Cin_g, kH, kW)
        x_sq = x.square()  # compute once, reused for p2 and x_grad_norm
        p2 = F.conv2d(x_sq, ones, stride=stride, padding=padding, groups=groups)
        # Use rsqrt instead of sqrt for pn — avoids a separate division later
        inv_pn = torch.rsqrt(p2 + eps)  # (B, groups, outH, outW)

        # Expand inv_pn to Cout by repeating within each group
        inv_pn_exp = inv_pn.repeat_interleave(Cout_g, dim=1)  # (B, Cout, outH, outW)

        # Weight norm per filter: (Cout,) — fuse with rsqrt
        w2 = W.square().sum(dim=(1, 2, 3))
        inv_wn = torch.rsqrt(w2 + eps)  # (Cout,)

        # denom = pn * wn → inv_denom = inv_pn * inv_wn
        inv_denom = inv_pn_exp * inv_wn.view(1, Cout, 1, 1)

        # corr = num / denom = num * inv_denom
        corr = (num * inv_denom).clamp_(-1.0 + 1e-4, 1.0 - 1e-4)  # in-place clamp

        d_asu_d_corr = (2.0 / torch.pi) * torch.rsqrt(1.0 - corr * corr)
        g = dy * d_asu_d_corr                           # (B, Cout, outH, outW)
        dL_dnum = g * inv_denom                         # (B, Cout, outH, outW)

        # dL/dpn per group: sum only over Cout_g channels in each group
        g_corr = g * corr  # reused for both dpn and dwn terms
        tmp = g_corr.view(B, groups, Cout_g, *corr.shape[2:]).sum(dim=2)
        # dL_dp2 = -0.5 * tmp * inv_pn^2
        # Chain rule: d(corr)/d(p2) = -0.5 * inv_pn^2 * corr  (not inv_pn^3;
        # tmp = sum_j(g_j * corr_j) already has corr which absorbs one inv_pn)
        dL_dp2 = -0.5 * tmp * inv_pn.pow(2)
        del tmp

        # backprop through p2 = grouped conv(x^2, ones)
        dL_dx2 = tgrad.conv2d_input(x.shape, ones, dL_dp2, stride=stride, padding=padding, groups=groups)
        del dL_dp2, ones
        # x_grad_norm = 2 * x * dL_dx2 — fuse multiply
        x_grad_norm = dL_dx2.mul_(2.0).mul_(x)  # in-place chain

        # grads via numerator conv
        x_grad_num = tgrad.conv2d_input(x.shape, W, dL_dnum, stride=stride, padding=padding, groups=groups)
        x_grad = x_grad_num.add_(x_grad_norm)  # in-place add
        del x_grad_num, x_grad_norm

        W_grad_num = tgrad.conv2d_weight(x, W.shape, dL_dnum, stride=stride, padding=padding, groups=groups)

        # wn term: dL_dwn = -s * inv_wn
        s = g_corr.sum(dim=(0, 2, 3))  # (Cout,)
        del g_corr
        dL_dwn = -s * inv_wn  # already have inv_wn
        W_grad_norm = W * (dL_dwn * inv_wn).view(Cout, 1, 1, 1)

        W_grad = W_grad_num.add_(W_grad_norm)  # in-place
        del W_grad_num, W_grad_norm

        # Return grads for: x, W, G, N, stride, padding, groups, eps, debug, chunk_N
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
        self.N = max(500, int(Cin_g * k * k * self.N_scale)) # Make N, N_scale times larger than the unfolded input feature size. or at least 1000. 
        
        self.weight = nn.Parameter(torch.randn(self.out_channels, Cin_g, k, k, **factory_kwargs))
        nn.init.normal_(self.weight, mean=0.0, std=0.05)
        self.bias = nn.Parameter(torch.zeros(self.out_channels, **factory_kwargs)) if bias else None
        # per-group G — pin memory for faster CPU→GPU DMA when G lives on CPU
        G = torch.sign(torch.randn(self.groups, self.N, K, **factory_kwargs))
        self.register_buffer("G", G)

        self.scale = nn.Parameter(torch.ones(self.out_channels, **factory_kwargs))

        # GPU cache for G (populated on first forward when G fits in VRAM)
        self._G_gpu: torch.Tensor | None = None

    @torch.no_grad()
    def resample_G(self):
        self.G.copy_(torch.randn_like(self.G))
        # Invalidate GPU cache — new G must be re-transferred
        self._G_gpu = None

    def extra_repr(self) -> str:
        # Match nn.Conv2d style: show core args + only non-defaults.
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
        # Use GPU-cached G if available and on the correct device;
        # otherwise use the registered buffer (may be CPU or GPU).
        G = self._G_gpu
        if G is None or G.device != x.device:
            if self.G.device == x.device:
                G = self.G
            else:
                G = self.G.to(x.device, non_blocking=True)
            # Cache on GPU if it fits (for subsequent forward calls)
            try:
                self._G_gpu = G
            except Exception:
                pass

        output = _Conv2d_GGD.apply(
            x, self.weight, G,
            self.N, self.stride, self.padding, self.groups,
            self.eps, self.debug, self.chunk_N,
        )

        # Fuse scale + bias into a single pass where possible
        output = output * self.scale.view(1, -1, 1, 1)

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        return output
