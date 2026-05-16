import random
import copy
import numpy as np
import os, sys
import json
import math
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from einops import rearrange, repeat
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from datetime import datetime
from copy import deepcopy
import collections

# Paths (notebooks/ -> project_root/src)


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data" / "imagenet"
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data" / "imagenet")).resolve()

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

print("PROJECT_ROOT:", PROJECT_ROOT)
print("SRC_ROOT:", SRC_ROOT)
print("DATA_DIR:", DATA_DIR)

if not hasattr(collections.abc, 'Sequence'):
    collections.abc.Sequence = collections.Sequence


# --- GGM Related Imports

from src.layers import GGMLinear,make_linear
from src.layers.normalization import RMSNorm
from src.layers.activations import NewGELU, OddGate
from src.utils.seed import make_G_from_seed


# Dataclasses


@dataclass
class ImageParams:
    width: int
    height: int
    in_channel: int


@dataclass
class ModelParameters:

    # stem
    patch_size: int

    # architecture
    depths: list          # e.g. [2,2,6,2]
    dims: list            # e.g. [64,128,256,512]
    heads: list           # e.g. [4,8]
    mlp_ratios: list      # e.g. [8,8,4,4]

    window_size: int = 7

    # regularization
    embed_dropout: float = 0.0
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0
    drop_path: float = 0.1

    # layer types
    layer_type: str = "fp"
    attn_type: str = "ggm"
    mlp_type: str = "ggm"
    
@dataclass
class Hyperparameters:
    batch_size: int
    out_classes: int
    epochs: int
    learning_rate: float
    weight_decay: float



#  
# Model
#  

import math
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


#   
# Helpers
#   

def trunc_normal_(tensor, std=0.02):
    return nn.init.trunc_normal_(tensor, std=std)


def norm_tokens(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # x: [B, C, H, W] -> apply LN over C
    B, C, H, W = x.shape
    return norm(x.permute(0, 2, 3, 1).reshape(B, H * W, C)).reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()


class RPReLU(nn.Module):
    """
    BHViT-style shifted PReLU for token tensors [..., C]
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.move1 = nn.Parameter(torch.zeros(hidden_size))
        self.prelu = nn.PReLU(hidden_size)
        self.move2 = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x - self.move1
        y = self.prelu(y.transpose(-1, -2)).transpose(-1, -2)
        y = y + self.move2
        return y


class LayerScale(nn.Module):
    def __init__(self, hidden_size: int, init_value: float = 0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(hidden_size) * init_value)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.alpha + self.bias


class DropPath(nn.Module):
    """
    timm-style DropPath
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


#   
# Factories
#   

def default_linear_factory(in_features: int, out_features: int, bias: bool = True) -> nn.Module:
    return nn.Linear(in_features, out_features, bias=bias)


def default_conv_factory(
    in_ch: int,
    out_ch: int,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    bias=True,
    groups=1,
) -> nn.Module:
    return nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        groups=groups,
    )


#   
# Config
#   

@dataclass
class BHViTConfig:
    img_size: int = 224
    in_chans: int = 3
    num_classes: int = 1000

    patch_size: int = 4

    depths: List[int] = None               # e.g. [2, 2, 6, 2] or [3, 4, 8, 4]
    dims: List[int] = None                 # e.g. [64, 128, 256, 512]
    heads: List[int] = None                # stage3-4 heads, e.g. [4, 8]
    mlp_ratios: List[int] = None           # e.g. [8, 8, 4, 4]

    drop: float = 0.0
    attn_drop: float = 0.0
    drop_path: float = 0.1
    layer_scale_init: float = 0.1
    window_size: int = 7

    use_stage_pos_embed: bool = True

    def __post_init__(self):
        if self.depths is None:
            self.depths = [2, 2, 6, 2]
        if self.dims is None:
            self.dims = [64, 128, 256, 512]
        if self.heads is None:
            self.heads = [4, 8]
        if self.mlp_ratios is None:
            self.mlp_ratios = [8, 8, 4, 4]


#   
# Patch stem and downsampling
#   

class PatchStem(nn.Module):
    """
    BHViT initial patch embedding: conv stride 4 -> 56x56 for 224 input
    """
    def __init__(self, cfg: BHViTConfig):
        super().__init__()
        dim = cfg.dims[0]
        self.proj = nn.Conv2d(cfg.in_chans, dim, kernel_size=cfg.patch_size, stride=cfg.patch_size, bias=True)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        grid = cfg.img_size // cfg.patch_size
        self.pos = nn.Parameter(torch.zeros(1, dim, grid, grid))
        trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, C, H, W]
        B, C, H, W = x.shape
        xt = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        xt = self.norm(xt)
        xt = self.act(xt)
        x = xt.transpose(1, 2).reshape(B, C, H, W).contiguous()
        x = x + self.pos
        return x


class DownsampleBHViT(nn.Module):
    """
    Cleaner BHViT-style downsample:
      - norm over input tokens
      - stride-2 conv branch
      - avgpool residual branch
      - residual channel expansion
      - norm + RPReLU
      - stage positional embedding
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        in_hw: int,
        conv_factory: Callable = default_conv_factory,
        use_stage_pos_embed: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_hw = in_hw
        self.out_hw = in_hw // 2

        self.norm0 = nn.LayerNorm(in_dim)
        self.move = nn.Parameter(torch.zeros(1, in_dim, 1, 1))
        self.proj = conv_factory(
            in_dim, out_dim, kernel_size=2, stride=2, padding=0, bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_dim)
        self.act = RPReLU(out_dim)

        self.use_stage_pos_embed = use_stage_pos_embed
        if use_stage_pos_embed:
            self.pos = nn.Parameter(torch.zeros(1, out_dim, self.out_hw, self.out_hw))
            trunc_normal_(self.pos, std=0.02)
        else:
            self.pos = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.in_hw and W == self.in_hw, f"Expected {self.in_hw}, got {(H,W)}"

        xt = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        xt = self.norm0(xt)
        x_norm = xt.transpose(1, 2).reshape(B, C, H, W).contiguous()

        residual = self.pool(x_norm)  # [B, C, H/2, W/2]
        out = self.proj(x_norm + self.move)  # [B, out_dim, H/2, W/2]

        # expand pooled residual along channel dim to match out_dim
        repeat_factor = self.out_dim // self.in_dim
        assert self.out_dim % self.in_dim == 0, "out_dim must be multiple of in_dim"
        residual = torch.cat([residual for _ in range(repeat_factor)], dim=1)

        B2, C2, H2, W2 = out.shape
        out_t = out.permute(0, 2, 3, 1).reshape(B2, H2 * W2, C2)
        res_t = residual.permute(0, 2, 3, 1).reshape(B2, H2 * W2, C2)

        out_t = self.norm(out_t) + res_t
        out_t = self.act(out_t)
        out = out_t.transpose(1, 2).reshape(B2, C2, H2, W2).contiguous()

        if self.pos is not None:
            out = out + self.pos
        return out


#   
# GC Token Mixer Block (stage 1-2)
#   

class LearnableBias2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return x + self.bias


class TokenMixer(nn.Module):
    """
    BHViT GC token mixer:
      3 depthwise-ish dilated conv branches (here standard grouped=1 by default)
    """
    def __init__(self, dim: int, conv_factory: Callable = default_conv_factory):
        super().__init__()
        self.move = LearnableBias2d(dim)


        self.conv1 = conv_factory(dim, dim, 3, padding=1, groups=dim)
        self.conv2 = conv_factory(dim, dim, 3, padding=3, dilation=3, groups=dim)
        self.conv3 = conv_factory(dim, dim, 3, padding=5, dilation=5, groups=dim)


        self.norm = nn.LayerNorm(dim)
        self.act1 = RPReLU(dim)
        self.act2 = RPReLU(dim)
        self.act3 = RPReLU(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.move(x)

        x1 = self.conv1(x).permute(0, 2, 3, 1).reshape(B, H * W, C)
        x1 = self.act1(x1)

        x2 = self.conv2(x).permute(0, 2, 3, 1).reshape(B, H * W, C)
        x2 = self.act2(x2)

        x3 = self.conv3(x).permute(0, 2, 3, 1).reshape(B, H * W, C)
        x3 = self.act3(x3)

        out = self.norm(x1 + x2 + x3)
        out = out.transpose(1, 2).reshape(B, C, H, W).contiguous()
        return out


class MLPBHViT(nn.Module):
    """
    BHViT-style token MLP:
      dense -> norm + expanded residual -> RPReLU
      dense -> norm + pooled residual -> RPReLU
    """
    def __init__(
        self,
        dim: int,
        mlp_ratio: int,
        linear_factory: Callable = default_linear_factory,
        drop: float = 0.0,
        layer_scale_init: float = 0.1,
        drop_path: float = 0.0,
    ):
        super().__init__()
        hidden = dim * mlp_ratio
        self.hidden = hidden
        self.dim = dim
        self.mlp_ratio = mlp_ratio

    
        self.fc1 = make_linear(
                "ggm",
                in_features=dim,
                out_features=hidden,
                k_bits_x=2,
                k_bits_w=2,
                N_factor=0.632,
                rho_cap=0.99,
                rho_eps=0,
                table_grid_size=1024, 
                soft_rho = False, 
            )
        self.move1 = nn.Parameter(torch.zeros(dim))
        self.norm1 = nn.LayerNorm(hidden)

        self.act1 = OddGate(alpha=1.0)


        self.fc2 = make_linear(
                "ggm",
                in_features=hidden,
                out_features=dim,
                k_bits_x=2,
                k_bits_w=2,
                N_factor=0.632,
                rho_cap=0.99,
                rho_eps=0,
                table_grid_size=1024, 
                soft_rho = False,
            )
        self.move2 = nn.Parameter(torch.zeros(hidden))
        self.norm2 = nn.LayerNorm(dim)
        self.act2 = RPReLU(dim)

        self.dropout = nn.Dropout(drop)
        self.layerscale = LayerScale(dim, layer_scale_init)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor):
    
        hidden = self.norm1(self.fc1(x + self.move1))
    
        residual_expand = torch.cat([x for _ in range(self.mlp_ratio)], dim=-1)
        hidden = self.act1(hidden + residual_expand)
    
        out = self.norm2(self.fc2(hidden + self.move2))
    
        B, N, H = hidden.shape
        hidden_grouped = hidden.view(B, N, self.mlp_ratio, self.dim)
        pooled = hidden_grouped.mean(dim=2)
    
        out = self.act2(out + pooled)
    
        out = self.dropout(out)
        out = self.layerscale(out)
        out = self.drop_path(out)
    
        return out


class GCLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: int,
        linear_factory: Callable = default_linear_factory,
        conv_factory: Callable = default_conv_factory,
        drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.gc = TokenMixer(dim, conv_factory=conv_factory)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBHViT(
            dim=dim,
            mlp_ratio=mlp_ratio,
            linear_factory=linear_factory,
            drop=drop,
            layer_scale_init=layer_scale_init,
            drop_path=drop_path,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        x1 = self.norm1(x.permute(0, 2, 3, 1).reshape(B, H * W, C))
        x1 = x1.transpose(1, 2).reshape(B, C, H, W).contiguous()
        x = x + self.gc(x1)

        x2 = self.norm2(x.permute(0, 2, 3, 1).reshape(B, H * W, C))
        x = x + self.mlp(x2).transpose(1, 2).reshape(B, C, H, W).contiguous()
        return x


#   
# Attention block (stage 3-4)
#   

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    # x: [B, H, W, C] -> [B*nW, ws*ws, C]
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return x


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int, B: int) -> torch.Tensor:
    # windows: [B*nW, ws*ws, C] -> [B, H, W, C]
    C = windows.shape[-1]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x


class TokenForAttention(nn.Module):
    """
    BHViT merged tokens:
      avgpool + maxpool over coarse windows, blended by learnable alpha
    """
    def __init__(self, dim: int, window_size: int = 7):
        super().__init__()
        self.window_size = window_size
        self.alpha = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.norm = nn.LayerNorm(dim)

    def merge_token(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        w = self.window_size
        avg = F.avg_pool2d(x, kernel_size=w, stride=w)
        mx = F.max_pool2d(x, kernel_size=w, stride=w)

        avg = avg.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])
        mx = mx.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])
        merged = self.alpha * mx + (1.0 - self.alpha) * avg
        return merged

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        w = self.window_size
        windows = window_partition(x.permute(0, 2, 3, 1), w)  # [B*nW, ws*ws, C]
        merged = self.merge_token(x)                           # [B, nW, C]
        nW = merged.shape[1]

        # merged_rep = merged.repeat_interleave(nW, dim=0)      # [B*nW, nW, C]
        merged_rep = merged.unsqueeze(1).expand(-1, nW, -1, -1).reshape(-1, nW, merged.shape[-1])
        token_all = torch.cat([windows, merged_rep], dim=1)   # [B*nW, ws*ws+nW, C]
        token_all = self.norm(token_all)
        return token_all, windows.shape[1], nW


class BHViTSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        linear_factory: Callable = default_linear_factory,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.token_fa = TokenForAttention(dim, window_size=window_size)

        self.moveq = nn.Parameter(torch.zeros(dim))
        self.movek = nn.Parameter(torch.zeros(dim))
        self.movev = nn.Parameter(torch.zeros(dim))

        self.moveq2 = nn.Parameter(torch.zeros(dim))
        self.movek2 = nn.Parameter(torch.zeros(dim))
        self.movev2 = nn.Parameter(torch.zeros(dim))

        self.q = make_linear(
                "ggm",
                in_features=dim,
                out_features=dim,
                k_bits_x=2,
                k_bits_w=2,
                N_factor=0.632,
                rho_cap=0.99,
                rho_eps=0,
                table_grid_size=1024, 
                soft_rho = False, 
            )
        self.k = make_linear(
                "ggm",
                in_features=dim,
                out_features=dim,
                k_bits_x=2,
                k_bits_w=2,
                N_factor=0.632,
                rho_cap=0.99,
                rho_eps=0,
                table_grid_size=1024, 
                soft_rho = False, 
            )
        self.v = make_linear(
                "ggm",
                in_features=dim,
                out_features=dim,
                k_bits_x=2,
                k_bits_w=2,
                N_factor=0.632,
                rho_cap=0.99,
                rho_eps=0,
                table_grid_size=1024, 
                soft_rho = False, 
            )

        self.normq = nn.LayerNorm(dim)
        self.normk = nn.LayerNorm(dim)
        self.normv = nn.LayerNorm(dim)

        self.rpreluq = RPReLU(dim)
        self.rpreluk = RPReLU(dim)
        self.rpreluv = RPReLU(dim)
        self.proj = make_linear(
                "ggm",
                in_features=dim,
                out_features=dim,
                k_bits_x=2,
                k_bits_w=2,
                N_factor=0.632,
                rho_cap=0.99,
                rho_eps=0,
                table_grid_size=1024, 
                soft_rho = False, 
            )
        self.norm_context = nn.LayerNorm(dim)
        self.rprelu_context = RPReLU(dim)
        # self.rprelu_context = OddGate(alpha=1.0)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # local/global fusion
        self.parm = nn.Parameter(torch.full((1, 1, 1, dim), 0.5))

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def window_reverse_high(self, high_tokens: torch.Tensor, H: int, W: int, B: int) -> torch.Tensor:
        # high_tokens: [B*nW, nW, C]
        nW = (H // self.window_size) * (W // self.window_size)
        coarse_h = H // self.window_size
        coarse_w = W // self.window_size

        x = high_tokens.view(B, nW, nW, self.dim)  # [B, repeated_windows, merged_tokens, C]
        x = x.mean(dim=1)                           # [B, nW, C]
        x = x.view(B, coarse_h, coarse_w, self.dim)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=(H, W), mode="nearest")
        x = x.permute(0, 2, 3, 1).contiguous()     # [B, H, W, C]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        token_all, split_dim, nW = self.token_fa(x)   # [B*nW, ws*ws+nW, C]
        mixed_q = self.normq(self.q(token_all + self.moveq)) + token_all
        mixed_k = self.normk(self.k(token_all + self.movek)) + token_all
        mixed_v = self.normv(self.v(token_all + self.movev)) + token_all

        mixed_q = self.rpreluq(mixed_q)
        mixed_k = self.rpreluk(mixed_k)
        mixed_v = self.rpreluv(mixed_v)

        q = mixed_q + self.moveq2
        k = mixed_k + self.movek2
        v = mixed_v + self.movev2


        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        context = torch.matmul(attn, v)  # [B*nW, h, L, d]
        context = context.permute(0, 2, 1, 3).contiguous().view(token_all.shape[0], token_all.shape[1], self.dim)

        context = self.norm_context(self.proj(context)) + mixed_q + mixed_k + mixed_v
        context = self.rprelu_context(context)
        context = self.proj_drop(context)

        local_out = context[:, :split_dim, :]    # [B*nW, ws*ws, C]
        high_out = context[:, split_dim:, :]     # [B*nW, nW, C]

        local_grid = window_reverse(local_out, self.window_size, H, W, B)  # [B, H, W, C]
        high_grid = self.window_reverse_high(high_out, H, W, B)             # [B, H, W, C]

        fused = local_grid * self.parm + high_grid * (1.0 - self.parm)
        fused = fused.permute(0, 3, 1, 2).contiguous()
        return fused


class BHViTAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int,
        window_size: int,
        linear_factory: Callable = default_linear_factory,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = BHViTSelfAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            linear_factory=linear_factory,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBHViT(
            dim=dim,
            mlp_ratio=mlp_ratio,
            linear_factory=linear_factory,
            drop=drop,
            layer_scale_init=layer_scale_init,
            drop_path=drop_path,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        x1 = self.norm1(x.permute(0, 2, 3, 1).reshape(B, H * W, C))
        x1 = x1.transpose(1, 2).reshape(B, C, H, W).contiguous()
        x = x + self.attn(x1)

        x2 = self.norm2(x.permute(0, 2, 3, 1).reshape(B, H * W, C))
        x = x + self.mlp(x2).transpose(1, 2).reshape(B, C, H, W).contiguous()
        return x



# Full model


class BHViT(nn.Module):
    def __init__(
        self,
        cfg: BHViTConfig,
        linear_factory: Optional[Callable] = None,
        conv_factory: Optional[Callable] = None,
    ):
        super().__init__()
        self.cfg = cfg
        linear_factory = linear_factory or default_linear_factory
        conv_factory = conv_factory or default_conv_factory

        dims = cfg.dims
        depths = cfg.depths
        mlp_ratios = cfg.mlp_ratios
        heads = cfg.heads

        # feature sizes:
        # stem: img/4
        # stage2 input after ds1: img/8
        # stage3 input after ds2: img/16
        # stage4 input after ds3: img/32
        hw0 = cfg.img_size // 4
        hw1 = cfg.img_size // 8
        hw2 = cfg.img_size // 16
        hw3 = cfg.img_size // 32

        total_depth = sum(depths)
        dpr = torch.linspace(0, cfg.drop_path, total_depth).tolist()
        dp_idx = 0

        self.stem = PatchStem(cfg)

        # stage 1: GC
        self.stage1 = nn.ModuleList([
            GCLayer(
                dim=dims[0],
                mlp_ratio=mlp_ratios[0],
                linear_factory=linear_factory,
                conv_factory=conv_factory,
                drop=cfg.drop,
                drop_path=dpr[dp_idx + i],
                layer_scale_init=cfg.layer_scale_init,
            )
            for i in range(depths[0])
        ])
        dp_idx += depths[0]

        self.down1 = DownsampleBHViT(
            in_dim=dims[0],
            out_dim=dims[1],
            in_hw=hw0,
            conv_factory=conv_factory,
            use_stage_pos_embed=cfg.use_stage_pos_embed,
        )

        # stage 2: GC
        self.stage2 = nn.ModuleList([
            GCLayer(
                dim=dims[1],
                mlp_ratio=mlp_ratios[1],
                linear_factory=linear_factory,
                conv_factory=conv_factory,
                drop=cfg.drop,
                drop_path=dpr[dp_idx + i],
                layer_scale_init=cfg.layer_scale_init,
            )
            for i in range(depths[1])
        ])
        dp_idx += depths[1]

        self.down2 = DownsampleBHViT(
            in_dim=dims[1],
            out_dim=dims[2],
            in_hw=hw1,
            conv_factory=conv_factory,
            use_stage_pos_embed=cfg.use_stage_pos_embed,
        )

        # stage 3: attention
        self.stage3 = nn.ModuleList([
            BHViTAttentionBlock(
                dim=dims[2],
                num_heads=heads[0],
                mlp_ratio=mlp_ratios[2],
                window_size=cfg.window_size,
                linear_factory=linear_factory,
                drop=cfg.drop,
                attn_drop=cfg.attn_drop,
                drop_path=dpr[dp_idx + i],
                layer_scale_init=cfg.layer_scale_init,
            )
            for i in range(depths[2])
        ])
        dp_idx += depths[2]

        self.down3 = DownsampleBHViT(
            in_dim=dims[2],
            out_dim=dims[3],
            in_hw=hw2,
            conv_factory=conv_factory,
            use_stage_pos_embed=cfg.use_stage_pos_embed,
        )

        # stage 4: attention
        self.stage4 = nn.ModuleList([
            BHViTAttentionBlock(
                dim=dims[3],
                num_heads=heads[1],
                mlp_ratio=mlp_ratios[3],
                window_size=cfg.window_size,
                linear_factory=linear_factory,
                drop=cfg.drop,
                attn_drop=cfg.attn_drop,
                drop_path=dpr[dp_idx + i],
                layer_scale_init=cfg.layer_scale_init,
            )
            for i in range(depths[3])
        ])

        self.norm = nn.LayerNorm(dims[3])
        self.head = nn.Linear(dims[3], cfg.num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)      # [B, C0, 56, 56]

        for blk in self.stage1:
            x = blk(x)
        x = self.down1(x)     # [B, C1, 28, 28]

        for blk in self.stage2:
            x = blk(x)
        x = self.down2(x)     # [B, C2, 14, 14]

        for blk in self.stage3:
            x = blk(x)
        x = self.down3(x)     # [B, C3, 7, 7]

        for blk in self.stage4:
            x = blk(x)

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.norm(x)
        x = x.mean(dim=1)
        return x


    def forward(self, x: torch.Tensor, return_features=False):

        features = []   #     collect block outputs
    
        x = self.stem(x)
    
        # ---- stage 1 ----
        for blk in self.stage1:
            x = blk(x)
            if return_features:
                features.append(x)
    
        x = self.down1(x)
    
        # ---- stage 2 ----
        for blk in self.stage2:
            x = blk(x)
            if return_features:
                features.append(x)
    
        x = self.down2(x)
    
        # ---- stage 3 ----
        for blk in self.stage3:
            x = blk(x)
            if return_features:
                features.append(x)
    
        x = self.down3(x)
    
        # ---- stage 4 ----
        for blk in self.stage4:
            x = blk(x)
            if return_features:
                features.append(x)
    
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.norm(x)
        x = x.mean(dim=1)
    
        logits = self.head(x)
    
        if return_features:
            return logits, features
        else:
            return logits







#  
# Data Handler
#  

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm


class DataHandler:
    def __init__(self, image_information, batch_size, data_dir, use_fake_data=False):
        self.img_info = image_information
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.use_fake_data = use_fake_data

        # ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]

    
    # Train transforms (DeiT FP baseline)
    
    def _train_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(
                self.img_info.width,
                # scale=(0.08, 1.0),
                scale=(0.08, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
            transforms.RandomErasing(p=0.25),
        ])

    
    # Validation transforms
    
    def _val_transform(self):
        return transforms.Compose([
            transforms.Resize(
                int(self.img_info.width * 256 / 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(self.img_info.width),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    
    # Dataloaders (single-GPU OR DDP-safe)
    
    def get_dataloaders(self):
        import torch.distributed as dist
        from torchvision import datasets

        use_ddp = dist.is_available() and dist.is_initialized()

        train_root = self.data_dir / "train"
        val_root = self.data_dir / "val"

        use_fake = (not train_root.exists()) or (not val_root.exists())

        if use_fake:
            if (not use_ddp) or dist.get_rank() == 0:
                print("    ImageNet not found — using FakeData for smoke test.")

            train_dataset = datasets.FakeData(
                size=1024,
                image_size=(3, self.img_info.height, self.img_info.width),
                num_classes=1000,
                transform=self._train_transform(),
            )

            val_dataset = datasets.FakeData(
                size=256,
                image_size=(3, self.img_info.height, self.img_info.width),
                num_classes=1000,
                transform=self._val_transform(),
            )
        else:
            train_dataset = datasets.ImageFolder(
                root=train_root,
                transform=self._train_transform(),
            )

            val_dataset = datasets.ImageFolder(
                root=val_root,
                transform=self._val_transform(),
            )

        if use_ddp:
            train_sampler = DistributedSampler(
                train_dataset,
                shuffle=True,
                drop_last=True,
            )
            val_sampler = DistributedSampler(
                val_dataset,
                shuffle=False,
                drop_last=False,
            )
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            # timeout=120,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            # timeout=120,
        )


        if (not use_ddp) or dist.get_rank() == 0:
            print("DataLoaders created successfully.")
            print(f"Training samples:   {len(train_dataset)}")
            print(f"Validation samples: {len(val_dataset)}")
            print(f"Per-GPU batch size: {self.batch_size}")
            print(f"DDP enabled:        {use_ddp}")
            print(f"Using FakeData:     {use_fake}")

        return train_loader, val_loader, train_sampler


#  
# Evaluator
#  


class Evaluator:
    def __init__(self, model, val_loader, device, output_dir):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir

    def evaluate(self, model_path):
        use_ddp = dist.is_available() and dist.is_initialized()

        # Only rank 0 evaluates in DDP
        if use_ddp and dist.get_rank() != 0:
            return None

        print("\n--- Starting ImageNet Validation ---")

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()


        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                preds = outputs.argmax(dim=1)

                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        top1 = 100.0 * total_correct / total_samples
        print(f"\nImageNet Top-1 Accuracy: {top1:.2f}%")

        results = {
            "top1_accuracy": top1,
            "num_samples": total_samples,
        }

        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "val_metrics.json"), "w") as f:
            json.dump(results, f, indent=2)

        return results

#  
#  Configuration 
#  

def save_run_config(
    path: str,
    *,
    model: nn.Module,
    trainer,
    hparams,
    mparams,
    img_info,
    optimizer,
    scheduler,
    train_loader,
    seed: int,
):
    # unwrap DDP safely
    model_to_inspect = model.module if hasattr(model, "module") else model

    is_ddp = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if is_ddp else 1

    cfg = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "seed": int(seed),
        },

        # ---------------------------
        # Distributed
        # ---------------------------
        "distributed": {
            "ddp": bool(is_ddp),
            "world_size": int(world_size),
        },

        # ---------------------------
        # Data
        # ---------------------------
        "data": {
            "dataset": "imagenet-1k",
            "image_size": img_info.width,
            "batch_size_per_gpu": train_loader.batch_size,
            "global_batch_size": train_loader.batch_size * world_size,
            "num_workers": train_loader.num_workers,
        },

        # ---------------------------
        # Model (architecture-level)
        # ---------------------------
        "model": {
            "name": model_to_inspect.__class__.__name__,
            "patch_size": mparams.patch_size,
            "depths": mparams.depths,
            "dims": mparams.dims,
            "heads": mparams.heads,
            "mlp_ratios": mparams.mlp_ratios,
            "window_size": mparams.window_size,
        },

        # ---------------------------
        # Training
        # ---------------------------
        "training": {
            "epochs": hparams.epochs,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "weight_decay": hparams.weight_decay,
            "label_smoothing": trainer.criterion.label_smoothing,
            "grad_clip": trainer.grad_clip,
            "use_amp": trainer.use_amp,
            "val_every": trainer.val_every,
        },

        # ---------------------------
        # Optimizer (fully reconstructable)
        # ---------------------------
        "optimizer": {
            "type": optimizer.__class__.__name__,
            "defaults": {
                k: v
                for k, v in optimizer.defaults.items()
                if isinstance(v, (int, float, str, bool, tuple))
            },
            "param_groups": [
                {
                    "lr": pg.get("lr"),
                    "weight_decay": pg.get("weight_decay", 0.0),
                }
                for pg in optimizer.param_groups
            ],
        },

        # ---------------------------
        # Scheduler
        # ---------------------------
        "scheduler": {
            "type": scheduler.__class__.__name__ if scheduler else None,
            "params": {},
        },

        # ---------------------------
        # Augmentation
        # ---------------------------
        "augmentation": {
            "mixup": {
                "enabled": trainer.use_mixup,
                "alpha": trainer.mixup_alpha,
                "prob": trainer.mixup_prob,
            },
            "cutmix": {
                "enabled": trainer.use_cutmix,
                "alpha": trainer.cutmix_alpha,
                "prob": trainer.cutmix_prob,
            },
        },

        # ---------------------------
        # EMA (future-safe)
        # ---------------------------
        "ema": {
            "enabled": False,
            "decay": None,
        },

        # ---------------------------
        # GGM layers (auto-discovered)
        # ---------------------------
        "ggm_layers": [],
    }

    # ---- scheduler params (serializable only) ----
    if scheduler is not None:
        for k, v in scheduler.__dict__.items():
            if isinstance(v, (int, float, str, bool)):
                cfg["scheduler"]["params"][k] = v

    # ---- GGM layer discovery ----
    for name, m in model_to_inspect.named_modules():
        if hasattr(m, "k_bits_x") and hasattr(m, "k_bits_w"):
            cfg["ggm_layers"].append({
                "name": name,
                "type": m.__class__.__name__,
                "k_bits_x": int(m.k_bits_x),
                "k_bits_w": int(m.k_bits_w),
                "N_factor": getattr(m, "base_N_factor", None),
                "N": getattr(m, "base_N", None),
                "rho_cap": getattr(m, "rho_cap", None),
                "soft_rho": getattr(m, "soft_rho", None),
            })

    # ---- write config ----
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)

#  
#  Training  
#  


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR


def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    B, _, H, W = x.size()
    index = torch.randperm(B, device=x.device)

    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def build_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def ddp_reduce_mean(value, device):
    """
    Average a scalar across processes.
    Safe to call in single-GPU (no-op).
    """
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)

    t = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t.item()




class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        train_sampler,
        epochs,
        lr,
        weight_decay,
        output_dir,
        warmup_epochs=5,
        label_smoothing=0.1,
        use_amp=True,
        val_every=5,
        grad_clip=0.0,
        *,
        mparams,
        img_info,
        teacher_model=None,
    ):


        
        # Detect DDP vs single-GPU
        
        self.is_ddp = dist.is_available() and dist.is_initialized()

        if self.is_ddp:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0

        
        # Hyper-Parameters
        

        self.hparams = Hyperparameters(
            batch_size=train_loader.batch_size,
            out_classes=None,
            epochs=epochs,
            learning_rate=lr,
            weight_decay=weight_decay,
        )
        self.mparams = mparams
        self.img_info = img_info
        self.warmup_epochs = warmup_epochs
        self.label_smoothing = label_smoothing
        self.optim_betas = (0.9, 0.98)



        
        # Device
        
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_sampler = train_sampler

        self.epochs = epochs
        self.output_dir = output_dir
        self.val_every = val_every
        self.grad_clip = grad_clip

        
        # Loss (DeiT baseline)
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        
        # Model
        
        model = model.to(self.device)

        if self.is_ddp:
            self.model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,   # <<<<<<<<<<------------------------------------
                static_graph=True,   #     prevents rebuild-buckets surprises\
            )
        else:
            self.model = model

        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model = self.teacher_model.to(self.device)
            self.teacher_model.eval()

        
        # Optimizer (separate rho_eps_log params)
        
        # eps_params = []
        # main_params = []
        
        # for name, p in self.model.named_parameters():
        #     if not p.requires_grad:
        #         continue
        
        #     if "rho_eps_log" in name:
        #         eps_params.append(p)
        #     else:
        #         main_params.append(p)
        
        # if self.rank == 0:
        #     total = sum(p.numel() for p in self.model.parameters())
        #     trainable = sum(p.numel() for p in main_params) + sum(p.numel() for p in eps_params)
        #     print(f"Optimizer will update {trainable}/{total} parameters")
        #     print(f"rho_eps params: {sum(p.numel() for p in eps_params)}")

        # self.optim = torch.optim.SGD(
        #         [
        #             {"params": main_params},
        #             {"params": eps_params, "lr": lr * 10},
        #         ],
        #         lr=lr,
        #         momentum=0.9,
        #         weight_decay=weight_decay,
        #         nesterov=True,
        #     )

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.rank == 0:
            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in trainable_params)
            print(f"Optimizer will update {trainable}/{total} parameters")

            for name, p in self.model.named_parameters():
                if p.numel() > 500000:
                    print(name, p.numel()/1e6)
        
        self.optim = torch.optim.AdamW(
            trainable_params,
            # self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=self.optim_betas,
        )



        # self.optim = torch.optim.SGD(
        #     trainable_params,
        #     lr=lr,
        #     momentum=0.9,
        #     weight_decay=weight_decay,
        #     nesterov=True,
        # )


        
        # Scheduler: warmup + cosine
        
        steps_per_epoch = len(self.train_loader)
        self.lr_sch = build_warmup_cosine_scheduler(
            self.optim,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )

        
        # AMP
        
        self.use_amp = bool(use_amp)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)


        
        # Mixup / CutMix (unchanged)
        
        self.use_mixup = True
        self.use_cutmix = True
        self.mixup_alpha = 0.8
        self.cutmix_alpha = 1.0
        self.mixup_prob = 0.5    # 0.8
        self.cutmix_prob = 0.5   # 1.0

        
        # Book-keeping
        
        if self.rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)
        self.best_val_acc = 0.0

        # ---- N-factor annealing schedule ----
        # Define as list of (epoch, multiplier)
        # Example: at epoch 0 use 2.5x, at 5 use 2.0x, at 10 use 1.5x, etc.
        self.n_anneal_schedule = [
            # (epoch, scale_multiplier)
            # Example:
            # (start_epoch, 1.0),
        ]
        
        self._last_applied_anneal_epoch = None



    def _train_one_epoch(self, epoch: int):
        self.model.train()

        # IMPORTANT: only for DDP
        if self.is_ddp and self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        loader = self.train_loader
        if self.rank == 0:
            loader = tqdm(
                self.train_loader,
                desc=f"Train [epoch {epoch+1}/{self.epochs}]",
                ncols=120,
                dynamic_ncols=True,
            )


        for step, (images, labels) in enumerate(loader):

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            r = np.random.rand()

            # ---- Mix selection ----
            do_mixup = self.use_mixup and (r < self.mixup_prob)
            do_cutmix = self.use_cutmix and (not do_mixup) and (
                r < self.mixup_prob + (1 - self.mixup_prob) * self.cutmix_prob
            )

            if do_mixup:
                images, y_a, y_b, lam = mixup_data(
                    images, labels, alpha=self.mixup_alpha
                )
            elif do_cutmix:
                images, y_a, y_b, lam = cutmix_data(
                    images, labels, alpha=self.cutmix_alpha
                )

            self.optim.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.float16):
                logits = self.model(images)

                if self.teacher_model is not None:
                    logits, student_feats = self.model(images, return_features=True)
                
                    with torch.no_grad():
                        teacher_logits, teacher_feats = self.teacher_model(images, return_features=True)
                else:
                    logits = self.model(images)
                    teacher_logits = None
                    student_feats = None
                    teacher_feats = None
            
                if do_mixup or do_cutmix:
                    ce_loss = (
                        lam * self.criterion(logits, y_a)
                        + (1 - lam) * self.criterion(logits, y_b)
                    )
                    hard_labels = labels
                else:
                    ce_loss = self.criterion(logits, labels)
                    hard_labels = labels
            
                loss = ce_loss  
                
                teacher_loss = torch.tensor(0.0, device=self.device)
                reg_loss = torch.tensor(0.0, device=self.device)
                block_loss = torch.tensor(0.0, device=self.device)
                
                model = self.model.module if self.is_ddp else self.model
   

                if teacher_logits is not None:

                    T = 2.0  
                
                    teacher_loss = F.kl_div(
                        F.log_softmax(logits / T, dim=-1),
                        F.softmax(teacher_logits / T, dim=-1),
                        reduction="batchmean"
                    ) * (T * T)

                    #     BLOCK DISTILLATION
                    if teacher_feats is not None:
                        for s, t in zip(student_feats[-2:], teacher_feats[-2:]):  # last 2 blocks only
                    
                            s_n = F.layer_norm(s, s.shape[-1:])
                            t_n = F.layer_norm(t, t.shape[-1:])
                    
                            block_loss += F.mse_loss(s_n, t_n)
                
                for m in model.modules():
                    if hasattr(m, "_reg_loss") and m._reg_loss is not None:
                        reg_loss = reg_loss + m._reg_loss
                
                lambda_teacher = 0.0   #     tune this (start 0.5–2) 

                lambda_block = 0.0  # start with 0.5–1.0 

                loss = loss + reg_loss + lambda_teacher * teacher_loss + lambda_block * block_loss
                
                if step == 0 and self.rank == 0:
                    print(f"CE loss: {ce_loss.item():.4f}")
                    print(f"Teacher loss: {teacher_loss.item():.4f}")
                    print(f"Block loss: {block_loss.item():.4f}")
                    
                
                    # weighted contributions
                    print(f"Weighted Teacher: {(lambda_teacher * teacher_loss).item():.4f}")
                    print(f"Weighted Block: {(lambda_block * block_loss).item():.4f}")
                    print(f"Weighted Regularizer: {(reg_loss).item():.4f}")
                
                    total_aux = (lambda_teacher * teacher_loss + lambda_block * block_loss).item()
                    print(f"Total aux loss: {total_aux:.4f}")
                
                    if ce_loss.item() > 0:
                        print(f"Teacher/CE ratio: {teacher_loss.item() / ce_loss.item():.6f}")
                        print(f"Block/CE ratio: {block_loss.item() / ce_loss.item():.6f}")
                        print(f"Aux/CE ratio: {total_aux / ce_loss.item():.6f}")
    
   

            self.scaler.scale(loss).backward()

            # Grad clip
            if self.grad_clip is not None and self.grad_clip > 0:
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.grad_clip
                )
                
            
            self.scaler.step(self.optim)
            self.scaler.update()
            

                        
            self.lr_sch.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == hard_labels).sum().item()
            total_samples += hard_labels.size(0)

            if self.rank == 0 and step % 200 == 0:
                with open(os.path.join(self.output_dir, "heartbeat.txt"), "w") as f:
                    f.write(datetime.now().isoformat())
            if self.rank == 0:
                loader.set_postfix(
                    loss=f"{loss.item():.3f}",
                    lr=f"{self.optim.param_groups[0]['lr']:.2e}",
                )



        
        # Metric reduction
        
        loss_mean = total_loss / max(1, len(self.train_loader))

        if self.is_ddp:
            loss_mean = ddp_reduce_mean(loss_mean, self.device)

            correct_t = torch.tensor(
                [total_correct], device=self.device, dtype=torch.float32
            )
            samples_t = torch.tensor(
                [total_samples], device=self.device, dtype=torch.float32
            )
            dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(samples_t, op=dist.ReduceOp.SUM)

            acc = 100.0 * (correct_t / samples_t).item()
        else:
            acc = 100.0 * total_correct / max(1, total_samples)

        return loss_mean, acc


    @torch.no_grad()
    def _validate(self, epoch: int):
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(self.val_loader, desc=f"Val   [epoch {epoch+1}]", disable=(self.rank != 0))
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.float16):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        loss_mean = total_loss / max(1, len(self.val_loader))

        if self.is_ddp:
            loss_mean = ddp_reduce_mean(loss_mean, self.device)

            correct_t = torch.tensor([total_correct], device=self.device, dtype=torch.float32)
            samples_t = torch.tensor([total_samples], device=self.device, dtype=torch.float32)
            dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(samples_t, op=dist.ReduceOp.SUM)
            acc = 100.0 * (correct_t / samples_t).item()
        else:
            # single-GPU smoke test
            acc = 100.0 * total_correct / max(1, total_samples)

        return loss_mean, acc

        
    @torch.no_grad()
    def report_rho_stats(self):
        # only rank 0 prints
        if self.rank != 0:
            return
    
        model = self.model.module if self.is_ddp else self.model
    
        print("\n--- Rho Exceed Stats ---")
        for name, m in model.named_modules():
            if hasattr(m, "_rho_total_sum") and m._rho_total_sum.item() > 0:
                pct = 100.0 * (m._rho_exceed_sum / m._rho_total_sum)
                print(f"{name} rho_exceed% = {pct.item():.2f}")
    
                # reset counters
                m._rho_exceed_sum.zero_()
                m._rho_total_sum.zero_()
                
    @torch.no_grad()
    def apply_n_annealing(self, epoch: int):
        if not self.n_anneal_schedule:
            return
        if self._last_applied_anneal_epoch == epoch:
            return
    
        model = self.model.module if self.is_ddp else self.model
        keys = ["q","k","v","proj","fc1","fc2"]
    
        for sched_epoch, spec in self.n_anneal_schedule:
            if epoch != sched_epoch:
                continue
    
            # spec can be a float (uniform) or dict (per-type)
            if isinstance(spec, (int, float)):
                scale_map = {k: float(spec) for k in keys}
            else:
                scale_map = {k: float(spec.get(k, 1.0)) for k in keys}  # default 1.0 = untouched
    
            if self.rank == 0:
                print(f"\n    Applying N annealing at epoch {epoch}: {scale_map}")
    
            layer_types = {k: [] for k in keys}
            num_touched = 0
    
            for name, m in model.named_modules():
                if not hasattr(m, "set_N_factor"):
                    continue
    
                layer_key = None
                for k in keys:
                    if name.endswith(f".{k}"):
                        layer_key = k
                        break
                if layer_key is None:
                    continue
    
                scale = scale_map[layer_key]
                if scale == 1.0:
                    continue  # untouched
    
                new_factor = float(m.base_N_factor) * scale
                m.set_N_factor(new_factor)
                layer_types[layer_key].append(new_factor)
                num_touched += 1
    
            if self.rank == 0:
                print(f"Touched {num_touched} layers")
                for k, vals in layer_types.items():
                    if vals:
                        print(f"  {k}: {vals[0]:.3f} (n={len(vals)})")
    
            self._last_applied_anneal_epoch = epoch
            break

    

    @torch.no_grad()
    def smooth_resample_G(self, epoch, alpha=0.9):
    
        if epoch == 0 or epoch % 10 != 0:
            return
    
        if self.rank == 0:
            base_seed = torch.randint(0, 2**31 - 1, (1,)).item()
        else:
            base_seed = 0
    
        if self.is_ddp:
            obj = [base_seed]
            dist.broadcast_object_list(obj, src=0)
            base_seed = obj[0]
    
        model = self.model.module if self.is_ddp else self.model
    
        for name, m in model.named_modules():
            if hasattr(m, "G"):
    
                layer_seed = (base_seed + abs(hash(name))) % (2**31 - 1)
    
                G_new = make_G_from_seed(
                    seed=layer_seed,
                    N=m.G.size(0),
                    d=m.G.size(1),
                    device=m.G.device,
                )
    
                #     smooth blend instead of replace
                m.G.mul_(alpha).add_(G_new, alpha=(1 - alpha))
    
                # optional: renormalize rows
                # m.G.div_(m.G.norm(dim=1, keepdim=True).clamp_min(1e-6))
    
        if self.rank == 0:
            print(f"    Smooth G refresh at epoch {epoch}")


    @torch.no_grad()
    def report_n_factors(self):
        if self.rank != 0:
            return
    
        model = self.model.module if self.is_ddp else self.model
    
        keys = ["q", "k", "v", "proj", "fc1", "fc2"]
        stats = {k: [] for k in keys}
    
        for name, m in model.named_modules():
            if not hasattr(m, "base_N_factor"):
                continue
    
            for k in keys:
                if name.endswith(f".{k}"):
    
                    # try to get current value
                    if hasattr(m, "current_N_factor"):
                        val = float(m.current_N_factor)
                    elif hasattr(m, "N_factor"):
                        val = float(m.N_factor)
                    else:
                        val = float(m.base_N_factor)
    
                    stats[k].append(val)
                    break
    
        print("\n N_factor stats:")

        for k in keys:
            if stats[k]:
                vals = stats[k]
                print(f"  {k:5s}: {vals[0]:.3f} (n={len(vals)})")
            
    def fit(self, model_path: str, run_seed: int, start_epoch: int = 0):

        # history = []
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        
        # Save run config ONCE (rank 0 only)
        
        if self.rank == 0:
            save_run_config(
                path=os.path.join(self.output_dir, "run_config.json"),
                model=self.model,
                trainer=self,
                hparams=self.hparams,
                mparams=self.mparams,
                img_info=self.img_info,
                optimizer=self.optim,
                scheduler=self.lr_sch,
                train_loader=self.train_loader,
                seed=run_seed,
            )
            
        for epoch in range(start_epoch, self.epochs):
            
            if self.rank == 0:
                print(f"\n--- Epoch {epoch+1}/{self.epochs} ---")

        
            # ---- N annealing ----
            # self.apply_n_annealing(epoch)
            # self.report_n_factors()


            # ------G resampling ------
            
            # self.smooth_resample_G(epoch)


            train_loss, train_acc = self._train_one_epoch(epoch)

            # Validate every N epochs (for speed)
            if (epoch % self.val_every == 0) or (epoch == self.epochs - 1):
                val_loss, val_acc = self._validate(epoch)
            else:
                val_loss, val_acc = float("nan"), float("nan")

            # ---- NEW: rho diagnostics ----
            # self.report_rho_stats()


            # Rank-0 logging + checkpointing
            if self.rank == 0:
                row = {
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                    "lr": float(self.optim.param_groups[0]["lr"]),
                }
            
                csv_path = os.path.join(self.output_dir, "training_history.csv")
            
                df = pd.DataFrame([row])
                df.to_csv(
                    csv_path,
                    mode="a",
                    header=not os.path.exists(csv_path),
                    index=False,
                )
            
                os.sync()

                model_to_save = self.model.module if self.is_ddp else self.model
                last_path = os.path.join(self.output_dir, "last.pth")
                
                torch.save({
                    "model": model_to_save.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "scheduler": self.lr_sch.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": float(self.best_val_acc),
                }, last_path)
                
                
                # Save BEST checkpoint (only when validation ran and improved)
                
                if (not math.isnan(val_acc)) and (val_acc > self.best_val_acc):
                    self.best_val_acc = float(val_acc)
                
                    torch.save({
                        "model": model_to_save.state_dict(),
                        "optimizer": self.optim.state_dict(),
                        "scheduler": self.lr_sch.state_dict(),
                        "scaler": self.scaler.state_dict(),
                        "epoch": epoch,
                        "best_val_acc": float(self.best_val_acc),
                    }, model_path)
                
                    print(f"    Saved best model: {self.best_val_acc:.2f}% -> {model_path}")


                print(
                    f"lr={self.optim.param_groups[0]['lr']:.3e} | "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}% | "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%"
                )

        if self.rank == 0:
            print(f"\nTraining finished. Best val acc: {self.best_val_acc:.2f}%")
        return None

    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


#  
# Freeze / Unfreeze Utilities (Globally Safe Version)
#  

def set_layer_trainable(model, layer_name: str, trainable: bool):
    """
    Freeze or unfreeze a specific module by exact module name.
    Works for both raw model and DDP-wrapped model.
    """

    # unwrap DDP if needed
    if hasattr(model, "module"):
        model = model.module

    module_dict = dict(model.named_modules())

    if layer_name not in module_dict:
        raise ValueError(f"Layer '{layer_name}' not found in model.")

    module = module_dict[layer_name]

    for param_name, p in module.named_parameters(recurse=True):

        full_name = f"{layer_name}.{param_name}"

        # always protect kp_log_scale
        if full_name.endswith("kernel.kp_log_scale"):
            p.requires_grad = False
        else:
            p.requires_grad = trainable

def set_layers_trainable(model, layer_names, trainable: bool):
    """
    Freeze or unfreeze a list of layers.
    """
    for name in layer_names:
        set_layer_trainable(model, name, trainable)

def freeze_all(model):
    if hasattr(model, "module"):
        model = model.module

    for p in model.parameters():
        p.requires_grad = False

def unfreeze_all(model):
    if hasattr(model, "module"):
        model = model.module

    for name, p in model.named_parameters():
        # keep kp_log_scale always frozen
        if name.endswith("kernel.kp_log_scale"):
            p.requires_grad = False
        else:
            p.requires_grad = True

def print_trainable_parameters(model):
    total = 0
    trainable = 0

    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    print(f"Trainable params: {trainable}/{total}")

######################################################


def main(
    num_runs, 
    master_seed, 
    test_name, 
    image_information, 
    model_parameters, 
    hyperparameters, 
    resume_from_run: str = None, 
    freeze_on_resume: bool = False,
    teacher_from_run: str = None,
    ):

    import torch.distributed as dist
    import os
    import json
    from datetime import datetime


    RESULTS_ROOT = "results"
    DATASET_NAME = "imagenet"

    # -----------------------------
    # Decide whether to use DDP
    # -----------------------------
    wants_ddp = ("WORLD_SIZE" in os.environ) and (int(os.environ["WORLD_SIZE"]) > 1)
    ddp_initialized_here = False

    if wants_ddp and (not dist.is_initialized()):
        ddp_setup()
        ddp_initialized_here = True

    is_ddp = dist.is_available() and dist.is_initialized()

    rank = dist.get_rank() if is_ddp else 0
    world_size = dist.get_world_size() if is_ddp else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if is_ddp else 0

    def barrier():
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            
    def optimizer_to(optim, device):
        for state in optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device, non_blocking=True)

    def set_fixed_rho_eps(base_model, alpha=0.1, use_weight_ratio=True, fixed_value=None, rank=0):
        for name, m in base_model.named_modules():
            if hasattr(m, "kernel") and hasattr(m.kernel, "rho_eps_log"):
                with torch.no_grad():
                    if fixed_value is not None:
                        rho_eps = torch.tensor(float(fixed_value), dtype=m.weight.dtype, device=m.weight.device)
                    elif use_weight_ratio:
                        W = m.weight
                        w_sq_mean = (W.pow(2).sum(dim=-1)).mean()
                        rho_eps = alpha * w_sq_mean
                    else:
                        raise ValueError("Either fixed_value must be set or use_weight_ratio=True")
    
                    rho_eps = rho_eps.clamp_min(1e-12)
                    m.kernel.rho_eps_log.data.fill_(torch.log(rho_eps))
    
                m.kernel.rho_eps_log.requires_grad_(False)
    
                if rank == 0:
                    print(f"{name}: fixed rho_eps = {rho_eps.item():.6f}")
    # -----------------------------
    # Helpful perf flags (4090)
    # -----------------------------
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    NUM_RUNS = num_runs
    MASTER_SEED = master_seed

    for i in range(NUM_RUNS):
        run_seed = i + MASTER_SEED
        Trainer.set_seed(seed=run_seed)

        if rank == 0:
            print(f"\n--- Starting Run {i+1}/{NUM_RUNS} (Seed: {run_seed}) ---")
            print(f"is_ddp={is_ddp} | world_size={world_size} | local_rank={local_rank}")

        # -------------------------------------------------
        # Run directory with synchronized timestamp
        # -------------------------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if is_ddp:
            obj = [timestamp] if rank == 0 else [None]
            dist.broadcast_object_list(obj, src=0)
            timestamp = obj[0]

        # --- NEW: name the new run folder differently if resuming ---
        if resume_from_run is not None:
            run_name = f"{resume_from_run}_resume_{timestamp}"
        else:
            run_name = f"{test_name}_run{i+1}_{timestamp}"
        
        run_output_dir = os.path.join(RESULTS_ROOT, DATASET_NAME, run_name)

        # run_name = f"{test_name}_run{i+1}_{timestamp}"
        # run_output_dir = os.path.join(RESULTS_ROOT, DATASET_NAME, run_name)

        if rank == 0:
            os.makedirs(run_output_dir, exist_ok=True)
            os.makedirs(os.path.join(run_output_dir, "evaluation_results"), exist_ok=True)

        barrier()

        local_model_path = os.path.join(run_output_dir, "best_model.pth")

        img_info = image_information
        mparams = model_parameters
        hparams = hyperparameters

        # -------------------------------------------------
        # Data
        # -------------------------------------------------
        data_handler = DataHandler(
            image_information=img_info,
            batch_size=hparams.batch_size,   # per-GPU batch size
            data_dir=DATA_DIR
        )

        train_loader, val_loader, train_sampler = data_handler.get_dataloaders()


        # -------------------------------------------------
        # Model
        # -------------------------------------------------
        
        cfg = BHViTConfig(
            img_size=img_info.width,
            in_chans=img_info.in_channel,
            num_classes=hparams.out_classes,
        
            patch_size=mparams.patch_size,
        
            depths=mparams.depths,
            dims=mparams.dims,
            heads=mparams.heads,
            mlp_ratios=mparams.mlp_ratios,
        
            drop=mparams.embed_dropout,
            attn_drop=mparams.attn_dropout,
            drop_path=mparams.drop_path,
        
            window_size=mparams.window_size,
        )
        
        base_model = BHViT(cfg)


        # -------------------------------------------------
        #     Optional frozen teacher model
        # -------------------------------------------------
        teacher_model = None
        
        if teacher_from_run is not None:
            teacher_cfg = BHViTConfig(
                img_size=img_info.width,
                in_chans=img_info.in_channel,
                num_classes=hparams.out_classes,
        
                patch_size=mparams.patch_size,
        
                depths=mparams.depths,
                dims=mparams.dims,
                heads=mparams.heads,
                mlp_ratios=mparams.mlp_ratios,
        
                drop=mparams.embed_dropout,
                attn_drop=mparams.attn_dropout,
                drop_path=mparams.drop_path,
        
                window_size=mparams.window_size,
        
            )
        
            teacher_model = BHViT(teacher_cfg)
            teacher_model = teacher_model.to(torch.device("cuda", local_rank))
        
            teacher_ckpt_path = os.path.join(
                RESULTS_ROOT,
                DATASET_NAME,
                teacher_from_run,
                "last.pth",
            )
        
            if not os.path.exists(teacher_ckpt_path):
                raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt_path}")
        
            if rank == 0:
                print(f"📘 Loading teacher from: {teacher_ckpt_path}")
        
            teacher_ckpt = torch.load(teacher_ckpt_path, map_location="cpu")
            missing, unexpected = teacher_model.load_state_dict(teacher_ckpt["model"], strict=False)
        
            if rank == 0:
                print("Teacher loaded.")
                print("Teacher missing keys:", missing)
                print("Teacher unexpected keys:", unexpected)
        
            for p in teacher_model.parameters():
                p.requires_grad = False
        
            teacher_model.eval()

        # ---- Enable L3 spread regularization ----
        for m in base_model.modules():
            if hasattr(m, "lambda_l3"):
                m.lambda_l3 = 0.0   #<-%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        # ---- DEBUG: confirm regularizer enabled ----
        if rank == 0:
            enabled = any(
                hasattr(m, "lambda_l3") and m.lambda_l3 > 0
                for m in base_model.modules()
            )
            print("L3 regularizer enabled:", enabled)

            
        start_epoch = 0
        checkpoint = None

        

        # m = base_model.transformer.layers[0].attn.q
        # print("kp_log_scale requires_grad:", m.kernel.kp_log_scale.requires_grad)
     
        # -------------------------------------------------
        # Optional: Load checkpoint from previous run
        # -------------------------------------------------
        if resume_from_run is None:
            set_fixed_rho_eps(
                base_model,
                alpha=0.0,              # or whatever you want
                use_weight_ratio=True,  # rho_eps = alpha * mean(||w||^2)
                fixed_value=None,       # set e.g. 1.0 if you want absolute fixed value
                rank=rank,
            )

        if resume_from_run is not None:
            ckpt_path = os.path.join(
                RESULTS_ROOT,
                DATASET_NAME,
                resume_from_run,
                "last.pth",   # <-- resume should use last.pth
            )
        
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
            if rank == 0:
                print(f"    Resuming from: {ckpt_path}")
        
            checkpoint = torch.load(ckpt_path, map_location="cpu")
        
            missing, unexpected = base_model.load_state_dict(checkpoint["model"], strict=False)  # <<<<<<<<<<<--------------------------------

            # ---- reset rho_eps for ablation ----
            # model = trainer.model.module if trainer.is_ddp else trainer.model
            

            
            if freeze_on_resume:
                freeze_all(base_model)
                # Unfreeze all q and k layers
                for name, module in dict(base_model.named_modules()).items():
                    if name.endswith(".fc1") or name.endswith(".q") or  name.endswith(".k") or  name.endswith(".proj"):
                        set_layer_trainable(base_model, name, True)
                print_trainable_parameters(base_model)
            if not freeze_on_resume:
                unfreeze_all(base_model)

            set_fixed_rho_eps(
                base_model,
                alpha=0.0,
                use_weight_ratio=True,
                fixed_value=None,
                rank=rank,
            )

            
            if rank == 0:
                print("Load complete.")
                print("Missing keys:", missing)
                print("Unexpected keys:", unexpected)




        # -------------------------------------------------
        # LR scaling (DeiT recipe)
        # lr = 0.0005 * (global_batch / 512)
        # -------------------------------------------------
        global_batch = hparams.batch_size * world_size  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<----------------------------------
        base_lr = 0.0005 * (global_batch / 512.0)

        if rank == 0:
            print(f"Per-GPU batch={hparams.batch_size} | Global batch={global_batch} | LR={base_lr:.6f}")

        # -------------------------------------------------
        # Trainer
        # -------------------------------------------------
        trainer = Trainer(
            model=base_model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            mparams=mparams, img_info=img_info,
            epochs=hparams.epochs,
            lr=base_lr,
            weight_decay=hparams.weight_decay,
            output_dir=run_output_dir,
            warmup_epochs=5,
            label_smoothing=0.1,
            use_amp=True,
            val_every=1,
            grad_clip=0.0,
            teacher_model=teacher_model, 
        )

        barrier()
        


        if resume_from_run is not None:

            start_epoch = int(checkpoint["epoch"]) + 1
            trainer.best_val_acc = float(checkpoint.get("best_val_acc", 0.0))
        
            ckpt_param_count = len(checkpoint["optimizer"]["param_groups"][0]["params"])
            current_param_count = len(trainer.optim.param_groups[0]["params"])
        
            
            # Detect optimizer types
            
            ckpt_pg0 = checkpoint["optimizer"]["param_groups"][0]
        
            ckpt_is_sgd  = ("momentum" in ckpt_pg0)
            ckpt_is_adam = ("betas" in ckpt_pg0)
        
            cur_is_sgd  = isinstance(trainer.optim, torch.optim.SGD)
            cur_is_adam = isinstance(trainer.optim, torch.optim.AdamW)
        
            restored = False
        
            
            # Restore optimizer ONLY if type matches
            
            if (not freeze_on_resume) and (ckpt_param_count == current_param_count):
        
                same_type = (ckpt_is_sgd and cur_is_sgd) or (ckpt_is_adam and cur_is_adam)
        
                if same_type:
                    # trainer.optim.load_state_dict(checkpoint["optimizer"])
                    try:
                        trainer.optim.load_state_dict(checkpoint["optimizer"])
                    except ValueError:
                        print("    Optimizer state incompatible, skipping load.")
                    optimizer_to(trainer.optim, trainer.device)
        
                    trainer.lr_sch.load_state_dict(checkpoint["scheduler"])
                    trainer.scaler.load_state_dict(checkpoint["scaler"])
        
                    restored = True
        
                    if rank == 0:
                        print("    Optimizer + scheduler state restored (matching type).")
                else:
                    if rank == 0:
                        print("    Checkpoint optimizer type != current optimizer type.")
                        print("    Starting with fresh optimizer.")
        
            else:
                if rank == 0:
                    print("    Optimizer param mismatch or freeze_on_resume=True.")
                    print("    Starting with fresh optimizer.")

        
            # 
            # # Rebase LR to new base_lr
            # 
            new_base_lr = base_lr
            old_base_lr = trainer.lr_sch.base_lrs[0]
            scale = new_base_lr / old_base_lr if old_base_lr != 0 else 1.0
        
            trainer.lr_sch.base_lrs = [b * scale for b in trainer.lr_sch.base_lrs]

            for pg in trainer.optim.param_groups:
        
                # Adjust LR
                if "initial_lr" in pg:
                    pg["initial_lr"] *= scale
                pg["lr"] *= scale
        
                # Force correct weight decay
                pg["weight_decay"] = hparams.weight_decay
        
                # Update betas only if Adam
                if "betas" in pg:
                    pg["betas"] = trainer.optim_betas


            
            # Apply NEW base_lr cleanly (no scaling)
            
            
            # # Set scheduler base LR
            # trainer.lr_sch.base_lrs = [base_lr for _ in trainer.lr_sch.base_lrs]
            
            # # Set optimizer param group LR directly
            # for pg in trainer.optim.param_groups:
            #     pg["lr"] = base_lr
            #     pg["weight_decay"] = hparams.weight_decay
            
            #     if "betas" in pg:
            #         pg["betas"] = trainer.optim_betas
        


            
            #     FIX: Align scheduler step position to resumed epoch
            
            steps_per_epoch = len(train_loader)
            global_step = start_epoch * steps_per_epoch
            
            trainer.lr_sch.last_epoch = global_step
            trainer.lr_sch._step_count = global_step
            
            # Recompute LR for this step
            trainer.lr_sch.step()
        
            
            # Safe logging
            
            if rank == 0:
                print("\nResume summary:")
                print("  computed base_lr:", base_lr)
                print("  scheduler base_lrs:", trainer.lr_sch.base_lrs)
                print("  scheduler last_epoch:", trainer.lr_sch.last_epoch)
        
                pg0 = trainer.optim.param_groups[0]
        
                if "betas" in pg0:
                    print("  BETAS:", pg0["betas"])
                if "momentum" in pg0:
                    print("  MOMENTUM:", pg0["momentum"])
        
                for i, pg in enumerate(trainer.optim.param_groups):
        
                    extras = []
                    if "betas" in pg:
                        extras.append(f"betas={pg['betas']}")
                    if "momentum" in pg:
                        extras.append(f"momentum={pg['momentum']}")
        
                    extra_str = " | " + " | ".join(extras) if extras else ""
        
                    print(
                        f"  Param group {i}: "
                        f"lr={pg['lr']:.6f} | "
                        f"weight_decay={pg['weight_decay']}"
                        f"{extra_str}"
                    )
        # -------------------------------------------------
        # N-factor annealing schedule
        # -------------------------------------------------

        trainer.n_anneal_schedule = [
            # (start_epoch+1, {"q": 1.0, "k": 1.0, "v": 1.3, "proj": 1.3, "fc1": 1.3, "fc2": 1.3}),
            # (start_epoch+4, {"q": 1.0, "k": 1.0, "v": 0.9, "proj": 0.9, "fc1": 1.0, "fc2": 1.0}),
            # (start_epoch+9, {"q": 1.0, "k": 1.0, "v": 1.0, "proj": 1.0, "fc1": 0.9, "fc2": 0.9}),
            ]
        # -------------------------------------------------
        # Train
        # -------------------------------------------------
        history = trainer.fit(
            model_path=local_model_path,
            run_seed=run_seed,
            start_epoch=start_epoch,
        )
        barrier()

        # -------------------------------------------------
        # Final evaluation (rank 0 only)
        # -------------------------------------------------
        if rank == 0:
            eval_model = trainer.model.module if getattr(trainer, "is_ddp", False) else trainer.model
            evaluator = Evaluator(
                model=eval_model,
                val_loader=val_loader,
                device=trainer.device,
                output_dir=os.path.join(run_output_dir, "evaluation_results"),
            )
            final_metrics = evaluator.evaluate(model_path=local_model_path)
            print("Final metrics:", final_metrics)

        barrier()

    # -----------------------------
    # Cleanup
    # -----------------------------
    if ddp_initialized_here:
        ddp_cleanup()
#  
# DDP utilities (torchrun)
#  
import os
import torch
import torch.distributed as dist

def ddp_setup():
    """
    Initialize torch.distributed using env vars set by torchrun.
    torchrun sets: RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    """
    if dist.is_available() and not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        torch.cuda.set_device(local_rank)


    # Always set device if CUDA is available
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)

def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# Entry point

if __name__ == "__main__":

    
    #  OPTION A: Single-GPU smoke test (use manually)
    
    # Uncomment this block ONLY when debugging locally
    #
    # main(
    #     num_runs=1,
    #     master_seed=100,
    #     test_name="smoke_test_fp_deit_tiny",
    #
    #     image_information=ImageParams(
    #         width=224,
    #         height=224,
    #         in_channel=3,
    #     ),
    #
    #     model_parameters=ModelParameters(
    #         patch_size=16,
    #         inner_dim=192,
    #         transformer_layers=12,
    #         num_head=3,
    #         embed_dropout=0.0,
    #         attn_dropout=0.0,
    #         mlp_dropout=0.0,
    #     ),
    #
    #     hyperparameters=Hyperparameters(
    #         batch_size=32,
    #         out_classes=1000,
    #         epochs=2,
    #         learning_rate=0.0,
    #         weight_decay=0.05,
    #     )
    # )


    # 🔹 OPTION B: ImageNet training (torchrun-safe)

    main(
        num_runs=1,
        master_seed=100,
        test_name="test_run",
        resume_from_run=None,
        teacher_from_run=None,
        image_information=ImageParams(
            width=224,
            height=224,
            in_channel=3,
        ),
    
        model_parameters=ModelParameters(
    
            patch_size=4,
    
            depths=[1,1,2,1],                     #[2,2,6,2], [3,4,8,4], [1,1,2,1]
            dims=[32,64,128,256],                 #[48,96,192,384],        #[64,128,256,512],[32,64,128,256]
            heads=[2,4],                          #[3,6],                  #[4,8],[2,4]
            mlp_ratios=[4,4,4,4],                #[4,4,4,4],                #[8,8,4,4],
    
            window_size=7,
    
            embed_dropout=0.0,
            attn_dropout=0.0,
            mlp_dropout=0.0,
            drop_path=0.0,
    
            attn_type="ggm",
            mlp_type="ggm",
        ),
    
        hyperparameters=Hyperparameters(
            batch_size=256,
            out_classes=1000,
            epochs=200,
            learning_rate=0.0,
            weight_decay=0.0,
        )
    )

# ----------
# # Entry point (torchrun-safe)
# ----------
# if __name__ == "__main__":

#     main(
#         num_runs=1,
#         master_seed=100,
#         test_name="smoke_test_fp_deit_tiny_ddp",

#         image_information=ImageParams(
#             width=224,
#             height=224,
#             in_channel=3,
#         ),

#         model_parameters=ModelParameters(
#             patch_size=16,
#             inner_dim=192,
#             transformer_layers=12,
#             num_head=3,
#             embed_dropout=0.0,
#             attn_dropout=0.0,
#             mlp_dropout=0.0,
#         ),

#         hyperparameters=Hyperparameters(
#             batch_size=32,        # 🔻 SMALL per-GPU batch (safe)
#             out_classes=1000,
#             epochs=2,             # 🔻 VERY short
#             learning_rate=0.0,    # computed internally
#             weight_decay=0.05,
#         )
#     )







