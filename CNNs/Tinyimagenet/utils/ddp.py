from __future__ import annotations

import os
import torch
import torch.distributed as dist
from typing import NamedTuple

class DDPEnv(NamedTuple):
    is_distributed: bool
    is_main: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

_DDP_ENV = None

def ddp_setup(peek: bool = False) -> DDPEnv:
    global _DDP_ENV
    if _DDP_ENV is not None and not peek:
        return _DDP_ENV
    
    is_distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    
    if is_distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        if not peek:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            torch.cuda.set_device(local_rank)
        
        device = torch.device(f"cuda:{local_rank}")
        is_main = (rank == 0)

    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        is_main = True

    env = DDPEnv(is_distributed, is_main, rank, local_rank, world_size, device)
    
    if not peek:
        _DDP_ENV = env
        
    return env


def is_main_process(is_distributed: bool) -> bool:
    if is_distributed:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return False
    return True


def all_reduce_sum(data, device):
    if not isinstance(data, torch.Tensor):
        data_tensor = torch.tensor(data, device=device)
    else:
        data_tensor = data
        
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(data_tensor, op=dist.ReduceOp.SUM)
    
    if isinstance(data, torch.Tensor):
        return data_tensor
    return data_tensor.item()


def barrier(device):
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()