

import math
import torch
import torch.nn as nn

from .quantizers import (
    ggm_table_bins_and_levels,
    ggm_quantize,
)
from ..utils.seed import make_G_from_seed


# ============================================================
# SAFE BVN PDF
# ============================================================

def bvn_pdf_safe(x, y, rho):
    """
    Boundary pdf term used to build K'(rho) from quantizer rectangles.
    Handles +/-inf corners by returning zero for non-finite coordinates.
    """
    x, y, rho = torch.broadcast_tensors(x, y, rho)
    out = torch.zeros_like(rho)

    finite = torch.isfinite(x) & torch.isfinite(y)
    if finite.any():
        xf, yf, rf = x[finite], y[finite], rho[finite]
        one_minus = (1.0 - rf * rf).clamp_min(1e-12)
        z = (xf * xf - 2.0 * rf * xf * yf + yf * yf) / one_minus
        out[finite] = torch.exp(-0.5 * z) / (2.0 * math.pi * torch.sqrt(one_minus))

    return out


def l3_spread_regularizer(W, eps=1e-6):
    W_hat = W / (W.norm(dim=-1, keepdim=True) + eps)
    return (W_hat.abs() ** 3).sum(dim=-1).mean()


# ============================================================
# EXACT K TABLE
# ============================================================

class ExactKTable(nn.Module):
    """
    Fixed-geometry table for

        K(rho)   = E[Q_kx^{s0x}(Zx) Q_kw^{s0w}(Zw)]
        K'(rho) = dK / d rho

    Runtime amplitude scales are not baked into this table.
    """

    def __init__(
        self,
        k_bits_x: int,
        k_bits_w: int,
        band: float,
        grid_size: int = 2048,
        table_s0_x: float = 1.0,
        table_s0_w: float = 1.0,
        build_device: str = "cpu",
        build_dtype: torch.dtype = torch.float64,
        force_K0_zero: bool = True,
    ):
        super().__init__()

        self.grid_size = int(grid_size)
        self.table_s0_x = float(table_s0_x)
        self.table_s0_w = float(table_s0_w)

        self.register_buffer("band_f", torch.tensor(float(band), dtype=torch.float32), persistent=False)

        ex, vx = ggm_table_bins_and_levels(
            k_bits_x, s0=self.table_s0_x, device=build_device, dtype=torch.float64
        )
        ew, vw = ggm_table_bins_and_levels(
            k_bits_w, s0=self.table_s0_w, device=build_device, dtype=torch.float64
        )

        lx, ux = ex[:-1], ex[1:]
        ly, uy = ew[:-1], ew[1:]

        LX, LY = torch.meshgrid(lx, ly, indexing="ij")
        UX, UY = torch.meshgrid(ux, uy, indexing="ij")
        VX, VW = torch.meshgrid(vx, vw, indexing="ij")
        Vprod = VX * VW

        rho_grid = torch.linspace(-band, band, self.grid_size, device=build_device, dtype=build_dtype)
        r = rho_grid.view(-1, 1, 1)

        dP = (
            bvn_pdf_safe(UX, UY, r)
            - bvn_pdf_safe(LX, UY, r)
            - bvn_pdf_safe(UX, LY, r)
            + bvn_pdf_safe(LX, LY, r)
        )

        Kp_grid = (dP * Vprod).sum(dim=(1, 2))

        Kp_f32 = Kp_grid.to(torch.float32)
        rho_f32 = rho_grid.to(torch.float32)

        dr = rho_f32[1:] - rho_f32[:-1]
        avg = 0.5 * (Kp_f32[1:] + Kp_f32[:-1])
        inc = avg * dr

        K_grid = torch.empty_like(rho_f32)
        K_grid[0] = 0.0
        K_grid[1:] = torch.cumsum(inc, dim=0)

        if force_K0_zero:
            i0 = int((self.grid_size - 1) / 2)
            K_grid = K_grid - K_grid[i0]

        self.register_buffer("rho_grid", rho_f32, persistent=False)
        self.register_buffer("Kp_grid", Kp_f32, persistent=False)
        self.register_buffer("K_grid", K_grid, persistent=False)

        inv_step = (self.grid_size - 1) / (2.0 * band)
        self.register_buffer("inv_step", torch.tensor(inv_step, dtype=torch.float32), persistent=False)

    def _interp(self, grid: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        band = self.band_f
        r = rho.clamp(-band, band)
        t = (r + band) * self.inv_step
        i0 = t.floor().to(torch.int64)
        i1 = (i0 + 1).clamp(max=self.grid_size - 1)
        w1 = t - i0.to(t.dtype)
        w0 = 1.0 - w1
        return w0 * grid[i0] + w1 * grid[i1]

    def K(self, rho: torch.Tensor) -> torch.Tensor:
        return self._interp(self.K_grid, rho)

    def Kp(self, rho: torch.Tensor) -> torch.Tensor:
        return self._interp(self.Kp_grid, rho)


# ============================================================
# Kernel backward surrogate
# ============================================================

class GGMKernel(nn.Module):
    """
    Backward surrogate for the MC forward.

    If a table exists, use table K'.
    If both sides are 1-bit and no table exists, use arcsin derivative.
    Otherwise use identity derivative.

    Runtime post-quantization scale product multiplies the slope:

        d/d rho [s_x s_w K(rho)] = s_x s_w K'(rho)
    """

    def __init__(
        self,
        k_bits_x: int,
        k_bits_w: int,
        rho_cap: float = 0.995,
        soft_rho: bool = False,
        table: ExactKTable | None = None,
        rho_eps: float = 0.0,
    ):
        super().__init__()
        self.k_bits_x = int(k_bits_x)
        self.k_bits_w = int(k_bits_w)
        self.max_bits = max(self.k_bits_x, self.k_bits_w)
        self.rho_cap = float(rho_cap)
        self.soft_rho = bool(soft_rho)
        self.table = table

        self.rho_eps_log = nn.Parameter(
            torch.tensor(math.log(max(float(rho_eps), 1e-12)), dtype=torch.float32)
        )

    @property
    def rho_eps(self):
        return torch.exp(self.rho_eps_log)

    def shape_rho(self, rho: torch.Tensor) -> torch.Tensor:
        if self.soft_rho:
            cap = self.rho_cap
            return cap * torch.tanh(rho / cap)
        return rho.clamp(-self.rho_cap, self.rho_cap)

    def slope(self, rho: torch.Tensor, scale_product: torch.Tensor | None = None) -> torch.Tensor:
        rho_s = self.shape_rho(rho)

        if self.table is not None:
            out = self.table.Kp(rho_s)
        elif self.max_bits == 1:
            denom = (1.0 - rho_s * rho_s).clamp_min(1e-6)
            out = (2.0 / math.pi) / torch.sqrt(denom)
        else:
            out = torch.ones_like(rho_s)

        if scale_product is not None:
            out = out * scale_product.to(device=out.device, dtype=out.dtype)

        return out


# ============================================================
# Projected quantization + post-quantization scale
# ============================================================

def _check_scale_policy(scale_policy: str) -> str:
    scale_policy = str(scale_policy).lower().strip()
    valid = {"none", "fixed", "learnable_mean"}
    if scale_policy not in valid:
        raise ValueError(f"scale_policy must be one of {sorted(valid)}, got {scale_policy!r}")
    return scale_policy


def _quantize_projected_base(z: torch.Tensor, k_bits: int, table_s0: float) -> torch.Tensor:
    """
    Geometry quantizer before runtime amplitude scaling.

    Important clean 1-bit policy:
      k_bits == 1 returns exactly +/-1, independent of table_s0.
      k_bits >= 2 uses the fixed table_s0 quantizer geometry.
    """
    if int(k_bits) == 1:
        return torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    return ggm_quantize(z, int(k_bits), s0=float(table_s0))


def quantize_projected_with_scale(
    z: torch.Tensor,
    k_bits: int,
    table_s0: float,
    scale_dim: int,
    scale_policy: str,
    log_scale_mult: torch.Tensor | None,
    fixed_scale: float,
    eps: float = 1e-6,
):
    """
    Returns q_scaled, scale.

    q0 = fixed geometry quantizer.
    q  = scale * q0.
    """
    if int(k_bits) >= 16:
        raise ValueError("Clean GGMLinear only supports finite-bit quantized sides.")

    scale_policy = _check_scale_policy(scale_policy)
    q0 = _quantize_projected_base(z, int(k_bits), table_s0=float(table_s0))

    scale_shape = list(z.shape)
    scale_shape[scale_dim] = 1

    if scale_policy == "none":
        scale = torch.ones(scale_shape, device=z.device, dtype=z.dtype)
    elif scale_policy == "fixed":
        scale = torch.ones(scale_shape, device=z.device, dtype=z.dtype) * float(fixed_scale)
    else:  # learnable_mean
        scale = z.detach().abs().mean(dim=scale_dim, keepdim=True).clamp_min(eps)
        if log_scale_mult is not None:
            scale = scale * torch.exp(log_scale_mult).to(device=z.device, dtype=z.dtype)

    return q0 * scale, scale


def combine_scale_product(
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
    batch_size: int,
    out_features: int,
    device,
    dtype,
):
    """Builds matrix s_x[b] s_w[o] used to multiply K'(rho)."""
    return (scale_x @ scale_w).to(device=device, dtype=dtype)


# ============================================================
# Custom autograd function
# ============================================================

class _GGMLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        W,
        G,
        k_bits_x: int,
        k_bits_w: int,
        kernel: GGMKernel,
        x_log_scale_mult,
        w_log_scale_mult,
        scale_policy: str,
        fixed_x_scale: float,
        fixed_w_scale: float,
        table_s0_x: float,
        table_s0_w: float,
        use_centering: bool = True,
        use_std_norm: bool = False,
    ):
        rho_eps = kernel.rho_eps
        std_eps = 1e-5
        scale_policy = _check_scale_policy(scale_policy)

        if int(k_bits_x) >= 16 or int(k_bits_w) >= 16:
            raise ValueError("Clean GGMLinear only supports finite-bit quantized sides.")

        if use_std_norm and not use_centering:
            raise ValueError("use_std_norm=True requires use_centering=True")

        # -------------------------
        # preprocessing
        # -------------------------
        if use_centering:
            x_c = x - x.mean(dim=-1, keepdim=True)
            W_c = W
        else:
            x_c = x
            W_c = W

        if use_std_norm:
            x_std = torch.sqrt(x_c.pow(2).mean(dim=-1, keepdim=True) + std_eps)
            W_std = torch.sqrt(W_c.pow(2).mean(dim=-1, keepdim=True) + std_eps)
            x_s = x_c / x_std
            W_s = W_c
        else:
            x_std = torch.ones_like(x_c[..., :1])
            W_std = torch.ones_like(W_c[..., :1])
            x_s = x_c
            W_s = W_c

        W_sq = W_s.pow(2).sum(dim=-1, keepdim=True)
        W_norm = torch.sqrt(W_sq + rho_eps + 1e-12)
        W_hat = W_s / W_norm

        x_sq = x_s.pow(2).sum(dim=-1, keepdim=True)
        x_norm = torch.sqrt(x_sq + 1e-12)
        x_hat = x_s / x_norm

        x_p = x_hat @ G.t()
        W_p = G @ W_hat.t()

        # -------------------------
        # MC forward estimator
        # -------------------------
        x_q, scale_x = quantize_projected_with_scale(
            x_p,
            k_bits=int(k_bits_x),
            table_s0=float(table_s0_x),
            scale_dim=-1,
            scale_policy=scale_policy,
            log_scale_mult=x_log_scale_mult,
            fixed_scale=float(fixed_x_scale),
        )

        W_q, scale_w = quantize_projected_with_scale(
            W_p,
            k_bits=int(k_bits_w),
            table_s0=float(table_s0_w),
            scale_dim=0,
            scale_policy=scale_policy,
            log_scale_mult=w_log_scale_mult,
            fixed_scale=float(fixed_w_scale),
        )

        rho_hat = (x_q @ W_q) / float(x_q.size(1))
        out = rho_hat

        scale_product = combine_scale_product(
            scale_x=scale_x,
            scale_w=scale_w,
            batch_size=x_q.size(0),
            out_features=W_q.size(1),
            device=x.device,
            dtype=x.dtype,
        )

        ctx.kernel = kernel
        ctx.use_centering = bool(use_centering)
        ctx.use_std_norm = bool(use_std_norm)
        ctx.k_bits_x = int(k_bits_x)
        ctx.k_bits_w = int(k_bits_w)
        ctx.scale_policy = scale_policy

        ctx.save_for_backward(
            x_s,
            W_s,
            x_norm,
            W_norm,
            x_std,
            W_std,
            scale_product,
            rho_hat,
        )

        rho_exact = x_hat @ W_hat.t()
        return out, rho_exact

    @staticmethod
    def backward(ctx, dy, _):
        x_s, W_s, x_norm, W_norm, x_std, W_std, scale_product, rho_hat = ctx.saved_tensors
        kernel: GGMKernel = ctx.kernel
        use_centering = ctx.use_centering
        use_std_norm = ctx.use_std_norm

        rho_eps = kernel.rho_eps

        x32 = x_s.float()
        W32 = W_s.float()
        x_std32 = x_std.float()

        w_sq = (W32 * W32).sum(dim=-1, keepdim=True)
        W_norm32 = torch.sqrt(w_sq + rho_eps + 1e-12)
        W_hat = W32 / W_norm32

        x_sq = (x32 * x32).sum(dim=-1, keepdim=True)
        x_norm32 = torch.sqrt(x_sq + 1e-12)
        x_hat = x32 / x_norm32

        rho = x_hat @ W_hat.t()
        rho = kernel.shape_rho(rho)

        slope = kernel.slope(rho, scale_product=scale_product.float()).float()
        U = dy.float() * slope

        dx_pre = (U @ W_hat) - (U * rho).sum(dim=1, keepdim=True) * x_hat
        dx_pre = dx_pre / x_norm32

        dW_pre = (U.t() @ x_hat) - (U * rho).sum(dim=0, keepdim=True).t() * W_hat
        dW_pre = dW_pre / W_norm32

        if use_std_norm:
            dx_pre = (
                dx_pre
                - dx_pre.mean(dim=-1, keepdim=True)
                - x32 * (dx_pre * x32).mean(dim=-1, keepdim=True)
            ) / x_std32

        if use_centering:
            dx_pre = dx_pre - dx_pre.mean(dim=-1, keepdim=True)

        dx = dx_pre.to(x_s.dtype)
        dW = dW_pre.to(W_s.dtype)

        # Manual rho_eps_log gradient preserved from current implementation.
        gW = -(
            ((U.t() @ x_hat) - (U * rho).sum(dim=0, keepdim=True).t() * W_hat)
            / W_norm32
            * W32
            / (W_norm32 ** 2)
        ).sum()
        drho_eps_log = gW * rho_eps

        if kernel.rho_eps_log.grad is None:
            kernel.rho_eps_log.grad = drho_eps_log.detach()
        else:
            kernel.rho_eps_log.grad += drho_eps_log.detach()

        # Learnable post-quantization amplitude scales.
        learnable = getattr(ctx, "scale_policy", "none") == "learnable_mean"
        kx = getattr(ctx, "k_bits_x", 16)
        kw = getattr(ctx, "k_bits_w", 16)

        if learnable and kx < 16:
            dx_log_scale_mult = (dy.float() * rho_hat.float()).sum().view(())
        else:
            dx_log_scale_mult = None

        if learnable and kw < 16:
            dw_log_scale_mult = (dy.float() * rho_hat.float()).sum(dim=0, keepdim=True)
        else:
            dw_log_scale_mult = None

        return (
            dx,                    # x
            dW,                    # W
            None,                  # G
            None,                  # k_bits_x
            None,                  # k_bits_w
            None,                  # kernel
            dx_log_scale_mult,     # x_log_scale_mult
            dw_log_scale_mult,     # w_log_scale_mult
            None,                  # scale_policy
            None,                  # fixed_x_scale
            None,                  # fixed_w_scale
            None,                  # table_s0_x
            None,                  # table_s0_w
            None,                  # use_centering
            None,                  # use_std_norm
        )


# ============================================================
# GGMLinear
# ============================================================

class GGMLinear(nn.Module):
    """
    Clean stochastic GGM layer.

    The forward computes a Monte Carlo estimator of the fixed-geometry kernel,
    optionally multiplied by post-quantization amplitude scales.

    scale_policy:
        none:
            every quantized side gets scale 1.

        fixed:
            every quantized side gets fixed_x_scale or fixed_w_scale.

        learnable_mean:
            every quantized side gets mean(abs(projected)).detach() * exp(log_scale).

    table_s0_x/table_s0_w are fixed quantizer/table geometry parameters for
    k_bits >= 2. For k_bits == 1, projected codes are exactly +/-1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k_bits_x: int,
        k_bits_w: int,
        rho_eps: float = 0.0,
        N_factor: float = 1.0,
        gain_init: float = 1.0,
        bias: bool = False,
        rho_cap: float = 0.995,
        soft_rho: bool = False,
        table_grid_size: int = 2048,
        table_bits_min: int = 2,
        table_bits_max: int = 5,
        table_s0_x: float = 1.0,
        table_s0_w: float = 1.0,
        use_centering: bool = True,
        use_std_norm: bool = False,
        scale_policy: str = "learnable_mean",
        fixed_x_scale: float = 1.0,
        fixed_w_scale: float = 1.0,
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.d = int(in_features)

        self.k_bits_x = int(k_bits_x)
        self.k_bits_w = int(k_bits_w)
        if self.k_bits_x >= 16 or self.k_bits_w >= 16:
            raise ValueError("Clean GGMLinear only supports finite-bit quantized sides. Use a separate layer for FP.")

        self.use_centering = bool(use_centering)
        self.use_std_norm = bool(use_std_norm)

        self.scale_policy = _check_scale_policy(scale_policy)
        self.fixed_x_scale = float(fixed_x_scale)
        self.fixed_w_scale = float(fixed_w_scale)

        self.table_s0_x = float(table_s0_x)
        self.table_s0_w = float(table_s0_w)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.gain = nn.Parameter(torch.ones(out_features) * float(gain_init))

        # Log multipliers receive gradients only when scale_policy == "learnable_mean".
        self.x_log_scale_mult = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.w_log_scale_mult = nn.Parameter(torch.zeros(1, out_features, dtype=torch.float32))

        self.lambda_l3 = 0.0
        self._reg_loss = None

        self.base_N_factor = float(N_factor)
        self.base_N = int(self.base_N_factor * self.d)
        if self.base_N <= 0:
            raise ValueError(f"N_factor gives invalid N={self.base_N}")

        self.G_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        G = make_G_from_seed(
            seed=self.G_seed,
            N=self.base_N,
            d=self.d,
            device=torch.device("cpu"),
        )
        self.register_buffer("G", G)

        table = None
        max_bits = max(self.k_bits_x, self.k_bits_w)
        if table_bits_min <= max_bits <= table_bits_max:
            # Public table_band removed: table range equals rho_cap.
            table = ExactKTable(
                k_bits_x=self.k_bits_x,
                k_bits_w=self.k_bits_w,
                band=float(rho_cap),
                grid_size=int(table_grid_size),
                table_s0_x=self.table_s0_x,
                table_s0_w=self.table_s0_w,
                build_device="cpu",
                build_dtype=torch.float64,
            )

        self.kernel = GGMKernel(
            k_bits_x=self.k_bits_x,
            k_bits_w=self.k_bits_w,
            rho_cap=float(rho_cap),
            soft_rho=bool(soft_rho),
            table=table,
            rho_eps=float(rho_eps),
        )
        self.table = table

        if self.table is not None:
            print(
                f"[GGMLinear clean] Exact K/K' table bits=({self.k_bits_x},{self.k_bits_w}) "
                f"s0=({self.table_s0_x},{self.table_s0_w}) "
                f"Kp_min={self.table.Kp_grid.min().item():.6f} "
                f"Kp_max={self.table.Kp_grid.max().item():.6f} "
                f"scale_policy={self.scale_policy}."
            )
        else:
            print(
                f"[GGMLinear clean] Table disabled for bits=({self.k_bits_x},{self.k_bits_w}); "
                f"using backward base kernel ({'arcsin' if max_bits == 1 else 'identity'}), "
                f"scale_policy={self.scale_policy}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])

        out, _rho = _GGMLinearFn.apply(
            x2,
            self.weight,
            self.G,
            self.k_bits_x,
            self.k_bits_w,
            self.kernel,
            self.x_log_scale_mult,
            self.w_log_scale_mult,
            self.scale_policy,
            self.fixed_x_scale,
            self.fixed_w_scale,
            self.table_s0_x,
            self.table_s0_w,
            self.use_centering,
            self.use_std_norm,
        )

        out = out * self.gain
        if self.bias is not None:
            out = out + self.bias

        if self.training and self.lambda_l3 > 0:
            self._reg_loss = self.lambda_l3 * l3_spread_regularizer(self.weight)
        else:
            self._reg_loss = None

        return out.reshape(*orig_shape[:-1], -1)

    @torch.no_grad()
    def active_scale_description(self):
        """Small debugging helper; does not compute batch-dependent mean scales."""
        return {
            "scale_policy": self.scale_policy,
            "fixed_x_scale": self.fixed_x_scale,
            "fixed_w_scale": self.fixed_w_scale,
            "table_s0_x": self.table_s0_x,
            "table_s0_w": self.table_s0_w,
            "one_bit_returns_pm_one": True,
            "x_has_runtime_scale": self.k_bits_x < 16,
            "w_has_runtime_scale": self.k_bits_w < 16,
            "N_factor": self.base_N_factor,
            "N": int(self.G.size(0)),
            "d": int(self.G.size(1)),
        }

def make_linear(layer_type, in_features, out_features, bias=False, **kwargs):
    """
    layer_type:
        - "fp"   : nn.Linear
        - "ggm"  : stochastic RF+quantized layer

    Shared kwargs (recommended to pass to both):
      k_bits_x, k_bits_w, rho_cap, soft_rho,
      table_band, table_grid_size, table_bits_min, table_bits_max,
      gain_init,  N_factor (accepted for parity)

    Notes:
      - mode is normalized so you can pass "dot-detached" or "dot_detached".
    """
    layer_type = str(layer_type).lower().strip()

    kwargs.setdefault("N_factor", 1.0)
    

    if layer_type == "fp":
        return nn.Linear(in_features, out_features, bias=bias)

    if layer_type == "ggm":
        return GGMLinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            **kwargs,
        )

    raise ValueError(f"Unknown layer_type={layer_type!r}. Use: fp | ggm")