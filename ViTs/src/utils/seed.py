# utils/seed.py
import random
import numpy as np
import torch
import math

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# def make_G_from_seed(seed: int, N: int, d: int, device):
#     """
#     ALWAYS generate on CPU, then move to device.
#     """
#     g = torch.Generator(device="cpu")
#     g.manual_seed(int(seed))

#     G_cpu = torch.randn(N, d, generator=g, device="cpu")
#     return G_cpu.to(device)


import torch

def make_G_from_seed(seed: int, N: int, d: int, device, normalize: bool = False):
    """
    Generate a Rademacher matrix G in {-1, +1}^{N x d} from a CPU seed,
    then move it to the requested device.

    Args:
        seed: random seed
        N: number of rows
        d: number of columns
        device: target device
        normalize: if True, returns G / sqrt(d)

    Returns:
        G: shape (N, d), dtype float32
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    # Sample 0/1 on CPU, map to -1/+1
    G_cpu = torch.randint(
        low=0,
        high=2,
        size=(N, d),
        generator=g,
        device="cpu",
        dtype=torch.int8,
    ).to(torch.float32)

    G_cpu = G_cpu.mul_(2.0).sub_(1.0)  # {0,1} -> {-1,+1}

    if normalize:
        G_cpu = G_cpu / (d ** 0.5)

    return G_cpu.to(device)


# def make_G_from_seed(seed: int, N: int, d: int, device):
#     """
#     Block-orthogonal Gaussian with antithetic pairing and proper scaling.
#     Works best when N >> d.
#     """
#     g = torch.Generator(device="cpu")
#     g.manual_seed(int(seed))

#     blocks = []
#     remaining = (N + 1) // 2   # build half, mirror later

#     while remaining > 0:
#         # Gaussian block
#         A = torch.randn(d, d, generator=g, device="cpu")

#         # QR → orthogonal rows
#         Q, _ = torch.linalg.qr(A)   # (d, d)

#         take = min(remaining, d)
#         blocks.append(Q[:take])

#         remaining -= take

#     G_half = torch.cat(blocks, dim=0)

#     # ---- antithetic pairing ----
#     G = torch.cat([G_half, -G_half], dim=0)[:N]

#     # ---- scale to match Gaussian variance ----
#     G = G * math.sqrt(d)

#     return G.to(device)

# import math
# import torch


# def _hadamard_matrix(n: int, device="cpu", dtype=torch.float32):
#     assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"
#     H = torch.tensor([[1.0]], device=device, dtype=dtype)
#     while H.shape[0] < n:
#         H = torch.cat([
#             torch.cat([H,  H], dim=1),
#             torch.cat([H, -H], dim=1),
#         ], dim=0)
#     return H


# def make_G_from_seed(seed: int, N: int, d: int, device):
#     """
#     Randomized Hadamard-style G.

#     Construction:
#       M = next power of 2 >= max(N, d)
#       H = M x M Hadamard / sqrt(M)
#       random row/col sign flips
#       random row/col permutations
#       take first N rows, first d cols
#     """
#     M = 1
#     while M < max(N, d):
#         M <<= 1

#     g = torch.Generator(device="cpu")
#     g.manual_seed(int(seed))

#     H = _hadamard_matrix(M, device="cpu", dtype=torch.float32) / math.sqrt(M)

#     row_signs = torch.randint(0, 2, (M,), generator=g, device="cpu", dtype=torch.int64)
#     row_signs = row_signs.float().mul_(2.0).sub_(1.0)

#     col_signs = torch.randint(0, 2, (M,), generator=g, device="cpu", dtype=torch.int64)
#     col_signs = col_signs.float().mul_(2.0).sub_(1.0)

#     row_perm = torch.randperm(M, generator=g, device="cpu")
#     col_perm = torch.randperm(M, generator=g, device="cpu")

#     H = H * row_signs.unsqueeze(1)
#     H = H * col_signs.unsqueeze(0)
#     H = H[row_perm][:, col_perm]

#     G_cpu = H[:N, :d].contiguous()
#     return G_cpu.to(device)
