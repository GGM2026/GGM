import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1) Single source of truth: quantizer spec
# ============================================================
def quantizer_spec(
    k_bits: int,
    s0: float,
    device="cpu",
    dtype=torch.float64,
):
    """
    Defines the exact clipped symmetric quantizer used everywhere.

    Returns:
      edges  : decision boundaries, shape [L+1]
      levels : reconstruction levels, shape [L]

    Quantizer:
      q(x) = s0 * round(clamp(x/s0, -1, 1) * Q) / Q
      Q = 2^(k_bits-1) - 1

    For k_bits == 1:
      q(x) = -s0 for x < 0, +s0 for x >= 0
    """
    if k_bits >= 16:
        raise ValueError("quantizer_spec only applies to finite quantizers (k_bits < 16).")
    if s0 <= 0:
        raise ValueError("s0 must be positive.")

    if k_bits == 1:
        edges = torch.tensor([-math.inf, 0.0, math.inf], device=device, dtype=dtype)
        levels = torch.tensor([-s0, +s0], device=device, dtype=dtype)
        return edges, levels

    Q = (2 ** (k_bits - 1)) - 1

    m = torch.arange(-Q, Q + 1, device=device, dtype=dtype)
    levels = s0 * (m / float(Q))

    k = torch.arange(-Q, Q, device=device, dtype=dtype)
    interior = s0 * ((k + 0.5) / float(Q))

    edges = torch.empty(levels.numel() + 1, device=device, dtype=dtype)
    edges[0] = -math.inf
    edges[1:-1] = interior
    edges[-1] = math.inf
    return edges, levels


# backward-compatible name for your table builder
def clipped_bins_and_levels(k_bits, s0=1.0, device="cpu", dtype=torch.float64):
    return quantizer_spec(k_bits=k_bits, s0=s0, device=device, dtype=dtype)


# ============================================================
# 2) Core forward quantizer from a given scale
# ============================================================
def quantize_core(x: torch.Tensor, k_bits: int, s0: torch.Tensor) -> torch.Tensor:
    """
    Exact same quantizer rule for both GGM and STE.
    Only the way s0 is chosen differs.

    s0 can be:
      - python float
      - scalar tensor
      - broadcastable tensor
    """
    if k_bits >= 16:
        return x

    if k_bits == 1:
        return torch.where(x >= 0, torch.as_tensor(s0, device=x.device, dtype=x.dtype), -torch.as_tensor(s0, device=x.device, dtype=x.dtype))

    Q = (2 ** (k_bits - 1)) - 1
    s0_t = torch.as_tensor(s0, device=x.device, dtype=x.dtype)
    x_scaled = (x / s0_t).clamp(-1.0, 1.0)
    return torch.round(x_scaled * Q) / Q * s0_t


# ============================================================
# 3) Same quantizer + STE wrapper
# ============================================================
def quantize_core_ste(
    x: torch.Tensor,
    k_bits: int,
    s0: torch.Tensor,
    clipped_ste: bool = True,
) -> torch.Tensor:
    x_q = quantize_core(x, k_bits, s0)

    if not clipped_ste:
        return x + (x_q - x).detach()

    s0_t = torch.as_tensor(s0, device=x.device, dtype=x.dtype)
    pass_through = (x.abs() <= s0_t).to(x.dtype)
    return x_q.detach() + (x - x.detach()) * pass_through


# ============================================================
# 4) Scale policies
# ============================================================
def fixed_scale(s0: float):
    """
    For GGM: use a fixed scalar chosen/tuned for the projected Gaussian signals.
    """
    return float(s0)


def amax_scale(x: torch.Tensor, dim=-1, keepdim=True, eps=1e-6):
    """
    For STE: signal-dependent scale from max abs value.
    """
    return x.abs().amax(dim=dim, keepdim=keepdim).clamp_min(eps)


def absmean_scale(x: torch.Tensor, dim=-1, keepdim=True, eps=1e-6):
    """
    Another STE option: signal-dependent scale from abs mean.
    """
    return x.abs().mean(dim=dim, keepdim=keepdim).clamp_min(eps)


# ============================================================
# 5) GGM-side helpers
# ============================================================
def ggm_quantize(x: torch.Tensor, k_bits: int, s0: float) -> torch.Tensor:
    """
    GGM forward quantization.
    Must use the same fixed s0 that your BVN table was built with.
    """
    return quantize_core(x, k_bits, s0)


def ggm_table_bins_and_levels(k_bits: int, s0: float, device="cpu", dtype=torch.float64):
    """
    Use this when building ExactKTable so the table matches GGM forward exactly.
    """
    return quantizer_spec(k_bits=k_bits, s0=s0, device=device, dtype=dtype)


# ============================================================
# 6) STE-side helpers
# ============================================================
def ste_quantize_activation(
    x: torch.Tensor,
    k_bits: int,
    scale_fn=amax_scale,
    clipped_ste: bool = True,
):
    s0 = scale_fn(x, dim=-1, keepdim=True)
    x_q = quantize_core_ste(x, k_bits, s0, clipped_ste=clipped_ste)
    return x_q, s0


def ste_quantize_weight(
    w: torch.Tensor,
    k_bits: int,
    scale_fn=amax_scale,
    clipped_ste: bool = True,
):
    # per-output-channel scale for Linear weights [out_features, in_features]
    s0 = scale_fn(w, dim=-1, keepdim=True)
    w_q = quantize_core_ste(w, k_bits, s0, clipped_ste=clipped_ste)
    return w_q, s0






# ============================================================
# 8) Optional debug checks
# ============================================================
def quantize_from_spec(x: torch.Tensor, edges: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    idx = torch.bucketize(x, edges[1:-1])
    return levels[idx]


def check_fixed_quantizer_consistency(k_bits: int, s0: float = 1.0):
    x = torch.linspace(-2.5 * s0, 2.5 * s0, 10001, dtype=torch.float64)
    edges, levels = quantizer_spec(k_bits, s0, device=x.device, dtype=x.dtype)
    q1 = quantize_from_spec(x, edges, levels)
    q2 = quantize_core(x, k_bits, s0)
    err = (q1 - q2).abs().max().item()
    print(f"k_bits={k_bits}, s0={s0}, max error = {err:.6e}")
    return err





