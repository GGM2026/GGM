# model.py
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from linearggm import LinearGGM
from layer_utils import LayerScale, OddGate, RMSNorm, RPReLU

class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.img_size = config.image_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.d_model = config.d_model

        self.proj = nn.Conv2d(config.in_channels, self.d_model, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

    def forward(self, x):
        bs = x.shape[0]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(bs, -1, -1)
        return torch.cat([cls, x], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        assert self.d_model % self.num_heads == 0
        self.head_dim = self.d_model // self.num_heads

        self.q = LinearGGM(self.d_model, self.d_model, centralize_x=False, centralize_w=False,)
        self.k = LinearGGM(self.d_model, self.d_model, centralize_x=False, centralize_w=False,)
        self.v = LinearGGM(self.d_model, self.d_model, centralize_x=False, centralize_w=False,)

        self.proj = LinearGGM(self.d_model, self.d_model, centralize_x=True, centralize_w=False,)

        self.attn_drop = nn.Dropout(config.attn_dropout)
        self.proj_drop = nn.Dropout(config.attn_dropout)

        self.scale = self.d_model * math.pi**2 / (4 * math.sqrt(self.head_dim))
        self.v_rms_log_alpha = nn.Parameter(torch.zeros(self.num_heads, 1, 1))

        self.attn_log_scale = nn.Parameter(torch.zeros(self.num_heads, 1, 1))
        self.eps = 1e-6

    def forward(self, x):
        bs, N, d_model = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.reshape(bs, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(bs, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(bs, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # q = q / (q.norm(dim=-1, keepdim=True)).add(self.eps)
        # k = k / (k.norm(dim=-1, keepdim=True)).add(self.eps)

        v_rms = v.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        v = v / v_rms
        v = v * self.v_rms_log_alpha.exp()

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale * self.attn_log_scale.exp()
        attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        out = attn @ v 
        out = out.transpose(1, 2).reshape(bs, N, d_model)

        out = self.proj(out)
        return self.proj_drop(out)

class FeedForward(nn.Module):
    def __init__(self, config, layer_scale_init: float = 0.1):
        super().__init__()
        self.hidden_size = config.ff_ratio * config.d_model

        self.ff1 = LinearGGM(
            config.d_model,
            self.hidden_size,
            centralize_x=False,
            centralize_w=False,
        )

        self.ff2 = LinearGGM(
            self.hidden_size,
            config.d_model,
            centralize_x=False,
            centralize_w=False,
        )

        self.act = RPReLU(self.hidden_size, channel_dim=-1, slope_init=0.25)
        self.drop = nn.Dropout(config.ff_dropout)

        # self.mlp_hidden_gain = nn.Parameter(torch.ones(1) * 2.0)

    def forward(self, x):
        x = self.ff1(x)
        x = self.act(x)

        # x = self.mlp_hidden_gain * x

        x = self.drop(x)
        x = self.ff2(x)
        x = self.drop(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_in_norm = RMSNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        # self.attn_out_norm = RMSNorm(config.d_model)
        self.attn_out_norm = nn.Identity()
        
        self.ff_in_norm = RMSNorm(config.d_model)
        self.ff = FeedForward(config)
        # self.ff_out_norm = RMSNorm(config.d_model)
        self.ff_out_norm = nn.Identity()

        self.alpha_attn = nn.Parameter(torch.tensor(0.05))
        self.alpha_ff = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        x = x + self.alpha_attn * self.attn_out_norm(self.attn(self.attn_in_norm(x)))
        x = x + self.alpha_ff * self.ff_out_norm(self.ff(self.ff_in_norm(x)))

        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(config.transformer_layers):
            layer = EncoderBlock(config)
            self.layers.append(layer)

    def forward(self, x):
        for encoder_layer in self.layers:
            x = encoder_layer(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_patches = (config.image_size // config.patch_size) ** 2
        self.patch_embed = PatchEmbedding(config)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.d_model))
        self.pos_drop = nn.Dropout(config.embed_dropout)

        self.transformer = Transformer(config)

        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.cls_token, std=0.02)

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (LinearGGM, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.transformer(x)
        x = self.norm(x[:, 0])
        x = self.head(x)
        return x

