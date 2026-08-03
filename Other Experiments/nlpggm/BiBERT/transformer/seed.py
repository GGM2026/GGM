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

##############################################################
## QR Generation
##############################################################
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

################################################################
## QR Generation Conv2dGGD
################################################################
import math
import torch


def make_G_conv2d_from_seed(
    seed,
    groups,
    N,
    cin_g,
    kh,
    kw,
    device,
):
    d = int(cin_g) * int(kh) * int(kw)

    group_blocks = []

    for group_idx in range(int(groups)):
        # This matches the old _make_grouped_G behavior.
        group_seed = int(seed + 10007 * group_idx)

        Gg = make_G_from_seed(
            seed=group_seed,
            N=int(N),
            d=int(d),
            device=torch.device("cpu"),
        )

        Gg = Gg.reshape(int(N), int(cin_g), int(kh), int(kw))
        group_blocks.append(Gg)

    return torch.stack(group_blocks, dim=0).to(device)