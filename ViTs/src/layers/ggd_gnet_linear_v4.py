# ============================================================
# Unified "Expectation-Parity" implementation for
#   - GGDLinear (stochastic RF + quantized estimator)
#   - GGMExpectationLinear (deterministic expectation kernel)
#
# Goal:
#   GGMExpectationLinear(x) == E[GGDLinear(x)]   (up to finite-N variance)
#
# Notes:
# - We do NOT hardcode arcsin inside GGD. For 1-bit, the kernel expectation is arcsin.
# - For 2..5 bits we use an exact BVN-based K'(rho) table and *integrate once* to get K(rho).
# - For >5 bits (or when table is disabled) we default to the continuous Gaussian limit kernel:
#       K(rho) = rho   (and K'(rho)=1)
#   which is the natural expectation target as quantization becomes fine / identity.
#
# Dependencies assumed to exist in your codebase:
#   layers.quantizers:
#       - quantize_kbit_affine
#       - clipped_bins_and_levels
#   layers.ggd_kernels_bvn (can be replaced by the ExactKTable below)
#   utils.seed.make_G_from_seed
#   utils.bitpack.pack_kbit
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizers import (
    ggd_table_bins_and_levels,
    ggd_quantize,
    ste_quantize_activation,
    ste_quantize_weight,
    amax_scale,
)
from ..utils.seed import make_G_from_seed
from ..utils.bitpack import pack_kbit


# ============================================================
# Global G registry
# ============================================================

class GlobalGRegistry:
    _registry = {}

    @classmethod
    def get_G(cls, d, N, device):
        key = (d, N)
        if key not in cls._registry:
            seed = 123456  # fixed global seed
            G = make_G_from_seed(seed=seed, N=N, d=d, device=device)
            cls._registry[key] = G
        return cls._registry[key]


# ============================================================
# SAFE BVN PDF (handles ±inf exactly)
# ============================================================
def bvn_pdf_safe(x, y, rho):
    """
    "PDF corner trick" used in your original K' construction.
    This is *not* a CDF; it's used to build K'(rho) via rectangle boundary terms.
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
    # row normalize
    W_hat = W / (W.norm(dim=-1, keepdim=True) + eps)
    return (W_hat.abs() ** 3).sum(dim=-1).mean()
    # return W_hat.abs().amax(dim=-1).mean()


# ============================================================
# EXACT K TABLE (build K'(rho) exactly, then integrate -> K(rho))
# ============================================================
class ExactKTable(nn.Module):
    """
    Builds:
      - Kp_grid(rho) : exact derivative K'(rho) (BVN rectangle boundary via pdf terms)
      - K_grid(rho)  : integrated expectation kernel K(rho), with K(0)=0

    Interpolation:
      - K(rho) and K'(rho) both via linear interpolation on the same rho grid.
    """

    def __init__(
        self,
        k_bits_x: int,
        k_bits_w: int,
        band: float,
        grid_size: int = 2048,
        build_device: str = "cpu",
        build_dtype: torch.dtype = torch.float64,
        force_K0_zero: bool = True,
    ):
        super().__init__()

        self.grid_size = int(grid_size)
        self.register_buffer("band_f", torch.tensor(float(band), dtype=torch.float32), persistent=False)

        # ---- quantizer geometry (clipped bins and representative levels) ----

        ex, vx = ggd_table_bins_and_levels(k_bits_x, s0=1.5, device=build_device, dtype=torch.float64)
        ew, vw = ggd_table_bins_and_levels(k_bits_w, s0=1.4, device=build_device, dtype=torch.float64)



        # bin edges (intervals)
        lx, ux = ex[:-1], ex[1:]
        ly, uy = ew[:-1], ew[1:]

        # grids
        LX, LY = torch.meshgrid(lx, ly, indexing="ij")
        UX, UY = torch.meshgrid(ux, uy, indexing="ij")
        VX, VW = torch.meshgrid(vx, vw, indexing="ij")
        Vprod = VX * VW

        # ---- rho grid ----
        rho_grid = torch.linspace(-band, band, self.grid_size, device=build_device, dtype=build_dtype)
        r = rho_grid.view(-1, 1, 1)

        # ---- EXACT K'(rho) grid construction (your original method) ----
        # dP here is a *boundary* combination of pdf values (not probability mass).
        dP = (
            bvn_pdf_safe(UX, UY, r)
            - bvn_pdf_safe(LX, UY, r)
            - bvn_pdf_safe(UX, LY, r)
            + bvn_pdf_safe(LX, LY, r)
        )

        Kp_grid = (dP * Vprod).sum(dim=(1, 2))  # shape: (grid_size,)

        # ---- Integrate K'(rho) -> K(rho) on the grid via cumulative trapezoid ----
        # We set K(0)=0 (by symmetry for odd-symmetric quantizers; good default).
        # Then integrate outward along the grid.
        # This makes K(rho) deterministic and consistent with K'(rho) used in backward.
        Kp_f32 = Kp_grid.to(torch.float32)
        rho_f32 = rho_grid.to(torch.float32)

        # trapezoid increments
        dr = (rho_f32[1:] - rho_f32[:-1])  # (grid_size-1,)
        avg = 0.5 * (Kp_f32[1:] + Kp_f32[:-1])
        inc = avg * dr  # (grid_size-1,)

        # cumulative from left
        K_grid = torch.empty_like(rho_f32)
        K_grid[0] = 0.0
        K_grid[1:] = torch.cumsum(inc, dim=0)

        # shift so that K(0)=0 exactly (optional but recommended)
        if force_K0_zero:
            # find nearest index to rho=0
            i0 = int((self.grid_size - 1) * (0.0 + band) / (2.0 * band))
            K_grid = K_grid - K_grid[i0]

        # store buffers (float32)
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
# Shared kernel logic (rho shaping + kernel selection + scaling)
# ============================================================
class GGDSharedKernel(nn.Module):
    """
    Encapsulates everything that must match between:
      - GGMExpectationLinear (deterministic)
      - GGDLinear (stochastic estimator)

    Provides:
      - forward_kernel(rho): returns K_bits(rho) with rho shaping and scaling
      - slope(rho): returns d/d rho K_bits(rho) with the same shaping and scaling

    Kernel selection policy:
      - if max_bits == 1: arcsin kernel (anonymousdieck)
      - if table exists (typically 2..5 bits): exact K from table
      - else: identity kernel (continuous Gaussian limit), K(rho)=rho
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
        self.rho_cap = float(rho_cap)
        self.soft_rho = bool(soft_rho)
        self.rho_eps_log = nn.Parameter(
            torch.tensor(math.log(max(rho_eps, 1e-12)), dtype=torch.float32)
        )

        self.table = table

        # choose base kernel type depending on bits
        self.max_bits = max(self.k_bits_x, self.k_bits_w)

        self.mixed_fp_x = (self.k_bits_x >= 16 and self.k_bits_w < 16)
        self.full_fp = (self.k_bits_x >= 16 and self.k_bits_w >= 16)
        
        if self.mixed_fp_x:
            c_q = mixed_kernel_gain_from_quantizer(self.k_bits_w, s0=1.0, dtype=torch.float64)
        else:
            c_q = torch.tensor(1.0, dtype=torch.float64)
        
        self.register_buffer("mixed_cq", c_q)

    @property
    def rho_eps(self):
        return torch.exp(self.rho_eps_log)

    def _shape_rho(self, rho: torch.Tensor) -> torch.Tensor:
        if self.soft_rho:
            cap = self.rho_cap
            return cap * torch.tanh(rho / cap)
        return rho.clamp(-self.rho_cap, self.rho_cap)

    def _base_K(self, rho: torch.Tensor) -> torch.Tensor:
        if self.mixed_fp_x:
            return self.mixed_cq.to(rho.dtype) * rho
    
        if self.max_bits == 1:
            return (2.0 / math.pi) * torch.asin(rho.clamp(-0.999, 0.999))
    
        return rho
    
    def _base_Kp(self, rho: torch.Tensor) -> torch.Tensor:
        if self.mixed_fp_x:
            return torch.ones_like(rho) * self.mixed_cq.to(rho.dtype)
    
        if self.max_bits == 1:
            denom = (1.0 - rho * rho).clamp_min(1e-6)
            return (2.0 / math.pi) / torch.sqrt(denom)
    
        return torch.ones_like(rho)

    def forward_kernel(self, rho: torch.Tensor) -> torch.Tensor:
        rho_s = self._shape_rho(rho)
        if self.mixed_fp_x:
            return self._base_K(rho_s)
        if self.table is not None:
            return self.table.K(rho_s)
        return self._base_K(rho_s)


    def slope(self, rho: torch.Tensor) -> torch.Tensor:
        rho_s = self._shape_rho(rho)
        if self.mixed_fp_x:
            return self._base_Kp(rho_s)
        if self.table is not None:
            return self.table.Kp(rho_s)
        return self._base_Kp(rho_s)

def mixed_kernel_gain_from_quantizer(
    k_bits_w: int,
    s0: float = 1.0,
    grid_lim: float = 8.0,
    grid_n: int = 200001,
    device="cpu",
    dtype=torch.float64,
):
    if k_bits_w >= 16:
        return torch.tensor(1.0, device=device, dtype=dtype)

    z = torch.linspace(-grid_lim, grid_lim, grid_n, device=device, dtype=dtype)
    dz = z[1] - z[0]
    pdf = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    qz = ggd_quantize(z, k_bits_w, s0=s0).to(dtype)
    return torch.sum(z * qz * pdf) * dz


class _GGDLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x, W, G,
        k_bits_x: int, k_bits_w: int,
        kernel: GGDSharedKernel,
        mode: str = "cosine",
        smooth_eps: float = 0.0,
        is_training: bool = True,
        use_centering: bool = True,
        use_std_norm: bool = True,
    ):
        rho_eps = kernel.rho_eps
        std_eps = 1e-5

        if use_std_norm and not use_centering:
            raise ValueError("use_std_norm=True requires use_centering=True")


        # --- center + std-normalize per row ---
        # -------------------------
        # preprocessing
        # -------------------------
        if use_centering:
            x_c = x - x.mean(dim=-1, keepdim=True)                 # (B,d)
            W_c = W - W.mean(dim=-1, keepdim=True)                 # (O,d)
        else:
            x_c = x
            W_c = W

        if use_std_norm:
            x_std = torch.sqrt(x_c.pow(2).mean(dim=-1, keepdim=True) + std_eps)
            W_std = torch.sqrt(W_c.pow(2).mean(dim=-1, keepdim=True) + std_eps)
            x_s = x_c / x_std
            W_s = W_c / W_std
        else:
            x_std = torch.ones_like(x_c[..., :1])
            W_std = torch.ones_like(W_c[..., :1])
            x_s = x_c
            W_s = W_c

        # --- then your usual norm-normalize ---

        mixed_fp1 = (k_bits_x >= 16 and k_bits_w == 1)

        W_sq = W_s.pow(2).sum(dim=-1, keepdim=True)
        W_norm = torch.sqrt(W_sq + rho_eps + 1e-12)
        W_hat = W_s / W_norm
        
        if mixed_fp1:
            x_hat = x_s
            x_norm = torch.ones_like(x_s[..., :1])
        else:
            x_sq = x_s.pow(2).sum(dim=-1, keepdim=True)
            x_norm = torch.sqrt(x_sq + 1e-12)
            x_hat = x_s / x_norm

        x_p = x_hat @ G.t()
        W_p = G @ W_hat.t()


        
        mixed_fp_x = (k_bits_x >= 16 and k_bits_w < 16)
        
        if is_training and smooth_eps > 0.0:
            if mixed_fp_x:
                sw = W_p.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
                W_p = W_p + (smooth_eps * sw) * torch.randn_like(W_p)
            else:
                sx = x_p.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
                sw = W_p.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
                x_p = x_p + (smooth_eps * sx) * torch.randn_like(x_p)
                W_p = W_p + (smooth_eps * sw) * torch.randn_like(W_p)
        
        if mixed_fp_x:
            W_q = ggd_quantize(W_p, k_bits_w, s0=1.0)
            rho_hat = (x_p @ W_q) / float(x_q.size(1))
             
        else:
            x_q = ggd_quantize(x_p, k_bits_x, s0=1.5)
            W_q = ggd_quantize(W_p, k_bits_w, s0=1.4)
            rho_hat = (x_q @ W_q) / float(x_q.size(1))
            
        
        out = rho_hat

        ctx.mode = mode
        if mode == "dot_detached":
            out = out * (x_norm.detach() * W_norm.detach().t())

        ctx.kernel = kernel
        ctx.use_centering = use_centering
        ctx.use_std_norm = use_std_norm
        
        rho_exact = x_hat @ W_hat.t()

        # save standardized tensors and downstream norms
        # ctx.save_for_backward(x_s, W_s, x_std, W_std, x_hat, W_hat, rho_exact, x_norm, W_norm)
        ctx.save_for_backward(x_s, W_s, x_norm, W_norm, x_std, W_std)
        ctx.mixed_fp1 = mixed_fp1

        return out, rho_exact

    @staticmethod
    def backward(ctx, dy, _):
        x_s, W_s, x_norm, W_norm, x_std, W_std = ctx.saved_tensors
        kernel: GGDSharedKernel = ctx.kernel
        use_centering = ctx.use_centering
        use_std_norm = ctx.use_std_norm
        mixed_fp1 = getattr(ctx, "mixed_fp1", False)

        if getattr(ctx, "mode", "cosine") == "dot_detached":
            dy = dy * (x_norm.detach() * W_norm.detach().t())

        rho_eps = kernel.rho_eps

        # standardized tensors
        x32 = x_s.float()
        W32 = W_s.float()
        x_std32 = x_std.float()
        W_std32 = W_std.float()

        # norm-normalization stage
        w_sq = (W32 * W32).sum(dim=-1, keepdim=True)
        W_norm32 = torch.sqrt(w_sq + rho_eps + 1e-12)
        W_hat = W32 / W_norm32

        
        if mixed_fp1:
            x_norm32 = torch.ones_like(x32[..., :1])
            x_hat = x32
        else:
            x_sq = (x32 * x32).sum(dim=-1, keepdim=True)
            x_norm32 = torch.sqrt(x_sq + 1e-12)
            x_hat = x32 / x_norm32

        rho = x_hat @ W_hat.t()
        rho = kernel._shape_rho(rho)
        g = kernel.slope(rho).float()

        U = dy.float() * g


        # grad wrt x_s, W_s
        if mixed_fp1:
            # x is raw FP, no cosine normalization on x
            dx_pre = U @ W_hat
        else:
            dx_pre = (U @ W_hat) - (U * rho).sum(dim=1, keepdim=True) * x_hat
            dx_pre = dx_pre / x_norm32

     
        
        dW_pre = (U.t() @ x_hat) - (U * rho).sum(dim=0, keepdim=True).t() * W_hat
        dW_pre = dW_pre / W_norm32

        # back through std normalization if enabled
        if use_std_norm:
            # if not mixed_fp1:
            dx_pre = (
                dx_pre
                - dx_pre.mean(dim=-1, keepdim=True)
                - x32 * (dx_pre * x32).mean(dim=-1, keepdim=True)
            ) / x_std32
        
            dW_pre = (
                dW_pre
                - dW_pre.mean(dim=-1, keepdim=True)
                - W32 * (dW_pre * W32).mean(dim=-1, keepdim=True)
            ) / W_std32

        # back through centering if enabled
        if use_centering:
            # if not mixed_fp1:
            dx_pre = dx_pre - dx_pre.mean(dim=-1, keepdim=True)
            dW_pre = dW_pre - dW_pre.mean(dim=-1, keepdim=True)

        dx = dx_pre.to(x_s.dtype)
        dW = dW_pre.to(W_s.dtype)

        gW = -( ((U.t() @ x_hat) - (U * rho).sum(dim=0, keepdim=True).t() * W_hat) / W_norm32 * W32 / (W_norm32**2) ).sum()
        drho_eps = gW
        drho_eps_log = drho_eps * rho_eps

        if kernel.rho_eps_log.grad is None:
            kernel.rho_eps_log.grad = drho_eps_log.detach()
        else:
            kernel.rho_eps_log.grad += drho_eps_log.detach()

        return dx, dW, None, None, None, None, None, None, None, None, None, None


# ============================================================
# Deterministic expectation-parity layer (proxy)
#   - exact cosine rho
#   - apply *the same* shared kernel as GGD uses
# ============================================================
class GGMLinear(nn.Module):
    """
    Deterministic "expectation" counterpart of GGDLinear.
    Same parameters (weight/gain/bias + kernel hyperparams).
    Forward:
      - exact cosine rho
      - shared kernel K_bits(rho)
      - gain/bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k_bits_x: int,
        k_bits_w: int,
        bias: bool = False,
        rho_cap: float = 0.995,
        soft_rho: bool = False,
        # NEW: match GGD interface
        mode: str = "cosine",      # "cosine" | "dot-detached"
        smooth_eps: float = 0.0,   # accepted for parity (unused)
        N_factor: float = 1.0,     # accepted for parity (unused)
        # table policy
        table_band: float | None = None,
        table_grid_size: int = 2048,
        table_bits_min: int = 2,
        table_bits_max: int = 5,
        gain_init: float = 1.0,
        rho_eps: float = 0.0,
        use_centering: bool = True,
        use_std_norm: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

        self.lambda_l3 = 0.0
        self._reg_loss = None

        self.gain = nn.Parameter(torch.ones(out_features) * float(gain_init))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.k_bits_x = int(k_bits_x)
        self.k_bits_w = int(k_bits_w)

        max_bits = max(self.k_bits_x, self.k_bits_w)


        # build table only in desired bit range

        mixed_fp_x = (self.k_bits_x >= 16 and self.k_bits_w < 16)

        table = None
        if (not mixed_fp_x):
            if table_band is None:
                table_band = float(rho_cap)
        
            max_bits = max(self.k_bits_x, self.k_bits_w)
            if (max_bits >= table_bits_min) and (max_bits <= table_bits_max):
                table = ExactKTable(
                    k_bits_x=self.k_bits_x,
                    k_bits_w=self.k_bits_w,
                    band=float(table_band),
                    grid_size=int(table_grid_size),
                    build_device="cpu",
                    build_dtype=torch.float64,
                )


        self.kernel = GGDSharedKernel(
            k_bits_x=self.k_bits_x,
            k_bits_w=self.k_bits_w,
            rho_cap=float(rho_cap),
            soft_rho=bool(soft_rho),
            table=table,
            rho_eps=rho_eps,
        )


        self.table = table
        
        mixed_fp_x = (self.k_bits_x >= 16 and self.k_bits_w < 16)
        
        if mixed_fp_x:
            print(
                f"[GGMLinear] Mixed mode: FP activations + {self.k_bits_w}-bit weights; "
                f"kernel = c_q * rho with c_q={self.kernel.mixed_cq.item():.6f}"
            )
        elif self.table is not None:
            print(
                f"[GGMLinear] Exact K/K' table bits=({self.k_bits_x},{self.k_bits_w}) "
                f"Kp_min={self.table.Kp_grid.min().item():.6f} "
                f"Kp_max={self.table.Kp_grid.max().item():.6f}"
            )
        else:
            print(
                f"[GGMLinear] Table disabled for bits=({self.k_bits_x},{self.k_bits_w}); "
                f"using base kernel ({'arcsin' if max_bits==1 else 'identity'})."
            )

        self.mode = str(mode)
        self.smooth_eps = float(smooth_eps)  # unused, parity only
        self.base_N_factor = float(N_factor) # unused, parity only

        self.register_buffer("_rho_exceed_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("_rho_total_sum", torch.tensor(0.0), persistent=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rho_eps = self.kernel.rho_eps
        std_eps = 1e-5   # MUST match _GGDLinearFn.forward
    
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
    
        # --- match _GGDLinearFn preprocessing exactly ---
        if self.use_centering:
            x_c = x2 - x2.mean(dim=-1, keepdim=True)
            W_c = self.weight - self.weight.mean(dim=-1, keepdim=True)
        else:
            x_c = x2
            W_c = self.weight
        
        if self.use_std_norm:
            x_std = torch.sqrt(x_c.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
            W_std = torch.sqrt(W_c.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
            x_s = x_c / x_std
            W_s = W_c / W_std
        else:
            x_s = x_c
            W_s = W_c
    
        x_sq = x_s.pow(2).sum(dim=-1, keepdim=True)
        W_sq = W_s.pow(2).sum(dim=-1, keepdim=True)
    
        x_norm = torch.sqrt(x_sq + 1e-12)
        W_norm = torch.sqrt(W_sq + rho_eps + 1e-12)
    
        rho = F.linear(x_s / x_norm, W_s / W_norm)
    
        # if self.training:
        #     cap = self.kernel.rho_cap
        #     exceed = (rho.abs() > cap).sum()
        #     total = rho.numel()
        #     self._rho_exceed_sum += exceed
        #     self._rho_total_sum += total

        # if self.training and torch.rand(()) < 0.001:  # sample rarely
        #     with torch.no_grad():
        #         x_norm_sq = x_sq.mean().item()
        #         w_norm_sq = W_sq.mean().item()
        #         eps = rho_eps.item()
        
        #         print(
        #             f"x||^2={x_norm_sq:.3f}  "
        #             f"w||^2={w_norm_sq:.3f}  "
        #             f"rho_eps={eps:.3f}  "
        #             f"x_ratio={eps/x_norm_sq:.4f}  "
        #             f"w_ratio={eps/w_norm_sq:.4f}"
        #         )
    
        out = self.kernel.forward_kernel(rho)
    
        if self.mode == "dot_detached":
            out = out * (x_norm.detach() * W_norm.t().detach())
    
        out = out * self.gain
        if self.bias is not None:
            out = out + self.bias
    
        if self.training and self.lambda_l3 > 0:
            self._reg_loss = self.lambda_l3 * l3_spread_regularizer(self.weight)
        else:
            self._reg_loss = None
    
        return out.reshape(*orig_shape[:-1], -1)







# ============================================================
# Stochastic GGD layer (RF estimator) using shared kernel
# ============================================================
class GGDLinear(nn.Module):
    """
    GGD layer:
      - estimates rho_hat with random projections + quantization
      - applies shared kernel K_bits(rho_hat)
      - uses stable backward with exact cosine rho for slope
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
        mode: str = "cosine",      # "cosine" or "dot_detached"
        smooth_eps: float = 0.0,
        # table policy
        table_band: float | None = None,
        table_grid_size: int = 2048,
        table_bits_min: int = 2,
        table_bits_max: int = 5,
        use_centering: bool = True,
        use_std_norm: bool = True,
    ):
        super().__init__()

        self.use_centering = use_centering
        self.use_std_norm = use_std_norm

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

        self.lambda_l3 = 0.0
        self._reg_loss = None


        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.gain = nn.Parameter(torch.ones(out_features) * float(gain_init))

        self.d = int(in_features)
        self.base_N_factor = float(N_factor)
        self.base_N = int(self.base_N_factor * self.d)

        self.k_bits_x = int(k_bits_x)
        self.k_bits_w = int(k_bits_w)

        self.mode = str(mode)
        self.smooth_eps = float(smooth_eps)

        # deterministic seed for G
        self.G_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        G = make_G_from_seed(
            seed=self.G_seed,
            N=self.base_N,
            d=self.d,
            device=torch.device("cpu"),
        )
        # G = GlobalGRegistry.get_G(self.d, self.base_N, self.weight.device)
        self.register_buffer("G", G)


        self.rho_cap = float(rho_cap)
        self.soft_rho = bool(soft_rho)
       

        max_bits = max(self.k_bits_x, self.k_bits_w)

        mixed_fp_x = (self.k_bits_x >= 16 and self.k_bits_w < 16)
        
        table = None
        if (not mixed_fp_x):
            if table_band is None:
                table_band = float(rho_cap)
        
            max_bits = max(self.k_bits_x, self.k_bits_w)
            if (max_bits >= table_bits_min) and (max_bits <= table_bits_max):
                table = ExactKTable(
                    k_bits_x=self.k_bits_x,
                    k_bits_w=self.k_bits_w,
                    band=float(table_band),
                    grid_size=int(table_grid_size),
                    build_device="cpu",
                    build_dtype=torch.float64,
                )


        self.kernel = GGDSharedKernel(
            k_bits_x=self.k_bits_x,
            k_bits_w=self.k_bits_w,
            rho_cap=float(rho_cap),
            soft_rho=bool(soft_rho),
            table=table,
            rho_eps=rho_eps,
        )

        
        
        # expose for parity/debug
        self.table = table
        
        # optional debug print
        mixed_fp_x = (self.k_bits_x >= 16 and self.k_bits_w < 16)
        
        if mixed_fp_x:
            print(
                f"[GGDLinear] Mixed mode: FP activations + {self.k_bits_w}-bit weights; "
                f"kernel = c_q * rho with c_q={self.kernel.mixed_cq.item():.6f}"
            )
        elif self.table is not None:
            print(
                f"[GGDLinear] Exact K/K' table bits=({self.k_bits_x},{self.k_bits_w}) "
                f"Kp_min={self.table.Kp_grid.min().item():.6f} "
                f"Kp_max={self.table.Kp_grid.max().item():.6f}"
            )
        else:
            print(
                f"[GGDLinear] Table disabled for bits=({self.k_bits_x},{self.k_bits_w}); "
                f"using base kernel ({'arcsin' if max_bits==1 else 'identity'})."
            )
                    


    @torch.no_grad()
    def resample_G(self):
        self.G_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        G = make_G_from_seed(
            seed=self.G_seed,
            N=self.G.size(0),
            d=self.G.size(1),
            device=self.G.device,
        )
        self.G.copy_(G)


    @torch.no_grad()
    def set_N_factor(self, N_factor: float):
        N = int(N_factor * self.d)
        if N == self.G.size(0):
            return
    
        G_new = make_G_from_seed(
            seed=self.G_seed,
            N=N,
            d=self.d,
            device=self.G.device,
        )
    
        self.G.resize_as_(G_new)
        self.G.copy_(G_new)
    
        self.base_N_factor = float(N_factor)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        # x2 = x.view(-1, orig_shape[-1])
        x2 = x.reshape(-1, orig_shape[-1])

        out, rho = _GGDLinearFn.apply(
            x2,
            self.weight,
            self.G,
            self.k_bits_x,
            self.k_bits_w,
            self.kernel,
            self.mode,
            self.smooth_eps,
            self.training,
            self.use_centering,
            self.use_std_norm,
        )

        # learned output gain + bias
        out = out * self.gain
        if self.bias is not None:
            out = out + self.bias


        #=========================================================
        #         ---- L3 spread regularizer ----
        #=========================================================
        if self.training and self.lambda_l3 > 0:
            self._reg_loss = self.lambda_l3 * l3_spread_regularizer(self.weight)
        else:
            self._reg_loss = None

        return out.reshape(*orig_shape[:-1], -1)




        @torch.no_grad()
        def export_inference_state(self):
            """
            Export inference state consistent with the current live GGD forward.
        
            Mixed mode:
              k_bits_x >= 16 and k_bits_w < 16
              -> runtime should keep x-projection full precision and quantize only weights
                 OR use precomputed effective_weight for fast large-batch inference.
        
            Symmetric mode:
              both sides quantized in projected space.
            """
            W = self.weight.detach()
            rho_eps = self.kernel.rho_eps
            std_eps = 1e-5
        
            # --- MUST match _GGDLinearFn preprocessing exactly ---
            W_c = W - W.mean(dim=-1, keepdim=True)                                # (O,d)
            W_std = torch.sqrt(W_c.pow(2).mean(dim=-1, keepdim=True) + std_eps)   # (O,1)
            W_s = W_c / W_std
        
            W_sq = W_s.pow(2).sum(dim=-1, keepdim=True)
            W_norm = torch.sqrt(W_sq + rho_eps + 1e-12)
            W_hat = W_s / W_norm
        
            # project
            W_p = self.G @ W_hat.t()   # (N, O)
        
            mixed_fp_x = (self.k_bits_x >= 16 and self.k_bits_w < 16)
        
            # use the SAME quantizer rule as live forward
            W_q = ggd_quantize(W_p, self.k_bits_w, s0=1.0)
        
            export = {
                "type": "linear",
                "G_seed": int(self.G_seed),
                "N": int(self.G.size(0)),
                "d": int(self.G.size(1)),
                "out_features": int(W.size(0)),
                "k_bits_x": int(self.k_bits_x),
                "k_bits_w": int(self.k_bits_w),
                "mixed_fp_x": bool(mixed_fp_x),
                "W_q": W_q.detach().cpu(),                # (N, O), already dequantized low-bit values
                "gain": self.gain.detach().cpu(),
                "bias": self.bias.detach().cpu() if self.bias is not None else None,
                "rho_cap": float(self.rho_cap),
                "soft_rho": bool(self.soft_rho),
                "rho_eps": self.kernel.rho_eps.detach().cpu(),
            }
        
            # Fast path for mixed mode:
            # (Gx)^T W_q / N == x^T (G^T W_q / N)
            if mixed_fp_x:
                W_eff = (self.G.t() @ W_q).t() / float(self.G.size(0))   # (O, d)
                export["effective_weight"] = W_eff.detach().cpu()
                export["mixed_cq"] = self.kernel.mixed_cq.detach().cpu()
        
            return export


# ============================================================
# Factory: fp | GGM | ggd
# ============================================================
def make_linear(layer_type, in_features, out_features, bias=False, **kwargs):
    """
    layer_type:
        - "fp"   : nn.Linear
        - "GGM" : deterministic expectation-parity layer
        - "ggd"  : stochastic RF+quantized layer

    Shared kwargs (recommended to pass to both):
      k_bits_x, k_bits_w, rho_cap, soft_rho,
      table_band, table_grid_size, table_bits_min, table_bits_max,
      gain_init, mode, smooth_eps, N_factor (accepted by GGM for parity)

    Notes:
      - mode is normalized so you can pass "dot-detached" or "dot_detached".
    """
    layer_type = str(layer_type).lower().strip()

    # defaults (safe for both)
    kwargs.setdefault("mode", "cosine")
    kwargs.setdefault("smooth_eps", 0.0)
    kwargs.setdefault("N_factor", 1.0)

    # normalize mode spelling
    mode = str(kwargs.get("mode", "cosine")).lower().strip()
    if mode in {"dot-detached", "dot_detached", "dotdetached"}:
        mode = "dot_detached"
    kwargs["mode"] = mode

    if layer_type == "fp":
        return nn.Linear(in_features, out_features, bias=bias)

    if layer_type == "GGM":
        return GGMLinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            **kwargs,
        )

    if layer_type == "ggd":
        return GGDLinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            **kwargs,
        )

    raise ValueError(f"Unknown layer_type={layer_type!r}. Use: fp | GGM | ggd")


class QLinearSTE(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        k_bits_x: int = 3,
        k_bits_w: int = 3,
        act_scale_fn=amax_scale,
        weight_scale_fn=amax_scale,
        clipped_ste: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.k_bits_x = k_bits_x
        self.k_bits_w = k_bits_w
        self.act_scale_fn = act_scale_fn
        self.weight_scale_fn = weight_scale_fn
        self.clipped_ste = clipped_ste

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q, _ = ste_quantize_activation(
            x,
            self.k_bits_x,
            scale_fn=self.act_scale_fn,
            clipped_ste=self.clipped_ste,
        )
        w_q, _ = ste_quantize_weight(
            self.weight,
            self.k_bits_w,
            scale_fn=self.weight_scale_fn,
            clipped_ste=self.clipped_ste,
        )
        return F.linear(x_q, w_q, self.bias)
