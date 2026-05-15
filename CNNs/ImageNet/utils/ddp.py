# utils/ddp.py
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DDPEnv:
    is_distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


def ddp_setup(backend: str | None = None) -> DDPEnv:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"

        dist.init_process_group(backend=backend)

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")

        return DDPEnv(True, rank, world_size, local_rank, device)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return DDPEnv(False, 0, 1, 0, device)


def is_main_process(is_distributed: bool) -> bool:
    return (not is_distributed) or (dist.is_available() and dist.is_initialized() and dist.get_rank() == 0)


@torch.no_grad()
def all_reduce_sum(value: float | int, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)

    t = torch.as_tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def barrier(device: torch.device | None = None) -> None:
    if dist.is_available() and dist.is_initialized():
        if dist.get_backend() == "nccl":
            if device is None:
                # fall back to current CUDA device
                device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else None
            if device is not None and device.type == "cuda":
                dist.barrier(device_ids=[device.index])
                return
        dist.barrier()


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
