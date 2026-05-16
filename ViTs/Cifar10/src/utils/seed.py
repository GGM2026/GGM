import random
import numpy as np
import torch
import math

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




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

    G_cpu = torch.randint(
        low=0,
        high=2,
        size=(N, d),
        generator=g,
        device="cpu",
        dtype=torch.int8,
    ).to(torch.float32)

    G_cpu = G_cpu.mul_(2.0).sub_(1.0)

    if normalize:
        G_cpu = G_cpu / (d ** 0.5)

    return G_cpu.to(device)
























