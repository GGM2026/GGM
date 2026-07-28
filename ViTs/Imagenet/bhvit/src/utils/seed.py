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


# import torch

# def make_G_from_seed(seed: int, N: int, d: int, device, normalize: bool = False):
#     """
#     Generate a Rademacher matrix G in {-1, +1}^{N x d} from a CPU seed,
#     then move it to the requested device.

#     Args:
#         seed: random seed
#         N: number of rows
#         d: number of columns
#         device: target device
#         normalize: if True, returns G / sqrt(d)

#     Returns:
#         G: shape (N, d), dtype float32
#     """
#     g = torch.Generator(device="cpu")
#     g.manual_seed(int(seed))

#     # Sample 0/1 on CPU, map to -1/+1
#     G_cpu = torch.randint(
#         low=0,
#         high=2,
#         size=(N, d),
#         generator=g,
#         device="cpu",
#         dtype=torch.int8,
#     ).to(torch.float32)

#     G_cpu = G_cpu.mul_(2.0).sub_(1.0)  # {0,1} -> {-1,+1}

#     if normalize:
#         G_cpu = G_cpu / (d ** 0.5)

#     return G_cpu.to(device)

###############################################################
###############################################################
def make_G_from_seed(seed: int, N: int, d: int, device):
    import math
    import torch

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    blocks = []
    remaining = int(N)

    while remaining > 0:
        A = torch.randn(d, d, generator=g, device="cpu")
        Q, R = torch.linalg.qr(A)

        signs = torch.sign(torch.diag(R))
        signs[signs == 0] = 1.0
        Q = Q * signs.view(1, -1)

        take = min(remaining, d)
        blocks.append(Q[:take])
        remaining -= take

    G = torch.cat(blocks, dim=0)

    # Match your original torch.randn(N, d) row scale.
    G = G * math.sqrt(d)

    return G.to(device)

###############################################################
###############################################################



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

# def make_G_from_seed(seed: int, N: int, d: int, device):
#     import math
#     import torch

#     seed = int(seed)
#     N = int(N)
#     d = int(d)

#     engine = torch.quasirandom.SobolEngine(
#         dimension=d,
#         scramble=True,
#         seed=seed,
#     )

#     # Avoid first degenerate Sobol point
#     engine.fast_forward(1)

#     U = engine.draw(N).to(dtype=torch.float32)

#     eps = 1e-6
#     U = U.clamp(eps, 1.0 - eps)

#     # Map Sobol points to quasi-Gaussian directions
#     Z = math.sqrt(2.0) * torch.erfinv(2.0 * U - 1.0)

#     # Centering is optional. I would NOT column-standardize here.
#     # Row-normalize to match block-ortho row norm exactly.
#     G = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-12)
#     G = G * math.sqrt(d)

#     return G.to(device)


# def make_G_from_seed(seed: int, N: int, d: int, device):
#     """
#     Global randomized Hadamard-like orthogonal projection.

#     For d not power-of-two:
#       - build Hadamard of size p = next_power_of_2(d)
#       - randomly select d columns
#       - randomly select N rows, with fresh blocks if N > p
#       - row-normalize to sqrt(d)

#     This keeps global mixing across all d coordinates.
#     No headwise/block-diagonal restriction.
#     """
#     import math
#     import torch

#     seed = int(seed)
#     N = int(N)
#     d = int(d)

#     g = torch.Generator(device="cpu")
#     g.manual_seed(seed)

#     def next_power_of_2(n):
#         return 1 << (int(n) - 1).bit_length()

#     def hadamard(n):
#         H = torch.ones(1, 1, dtype=torch.float32)
#         while H.size(0) < n:
#             H = torch.cat(
#                 [
#                     torch.cat([H, H], dim=1),
#                     torch.cat([H, -H], dim=1),
#                 ],
#                 dim=0,
#             )
#         return H

#     p = next_power_of_2(d)
#     H = hadamard(p)  # H H^T = p I

#     blocks = []
#     remaining = N

#     while remaining > 0:
#         # Random column sign flips: H D
#         signs = torch.randint(
#             low=0,
#             high=2,
#             size=(p,),
#             generator=g,
#             device="cpu",
#             dtype=torch.int64,
#         ).float()
#         signs = signs * 2.0 - 1.0
#         Hb = H * signs.view(1, -1)

#         # Randomly choose d columns: global mixing subset
#         col_perm = torch.randperm(p, generator=g, device="cpu")
#         cols = col_perm[:d]

#         # Randomly choose rows
#         row_perm = torch.randperm(p, generator=g, device="cpu")
#         take = min(remaining, p)
#         rows = row_perm[:take]

#         G_block = Hb[rows][:, cols]  # [take, d]

#         # Row-normalize to match Gaussian/QR row norm sqrt(d)
#         G_block = G_block / G_block.norm(dim=1, keepdim=True).clamp_min(1e-12)
#         G_block = G_block * math.sqrt(d)

#         blocks.append(G_block)
#         remaining -= take

#     G = torch.cat(blocks, dim=0)[:N]

#     return G.to(device)