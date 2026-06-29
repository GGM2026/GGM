from src.layers import GGDLinear, QLinearSTE, OddGate
from src.layers.normalization import RMSNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class PatchEmbedding(nn.Module):
    def __init__(self, mparams, hparams, img_info):
        super(PatchEmbedding, self).__init__()
        self.patch_size = mparams.patch_size
        self.img_size = img_info.width
        self.num_patches = (self.img_size//self.patch_size) * (self.img_size//self.patch_size)
        self.D = mparams.inner_dim
        self.patch_embed = nn.Conv2d(
            in_channels=img_info.in_channel,
            out_channels=self.D,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        self.cls_token = nn.Parameter(torch.rand(1,1,self.D))

    def forward(self, x):
        b = x.shape[0]
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = self.patch_embed(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = torch.cat((cls_token, x), dim=1)
        return x
    
class MHA(nn.Module):
    def __init__(self, mparams, hparams):
        super().__init__()
        self.D = mparams.inner_dim
        self.num_head = mparams.num_head
        assert self.D % self.num_head == 0
        self.head_size = self.D // self.num_head
        self.all_head_size = self.D

        self.query = GGDLinear(self.D, self.all_head_size, k_bits_x=mparams.k_bits_x, k_bits_w=mparams.k_bits_w, N_factor=mparams.n_factor, rho_cap=mparams.rho_cap)
        self.key   = GGDLinear(self.D, self.all_head_size, k_bits_x=mparams.k_bits_x, k_bits_w=mparams.k_bits_w, N_factor=mparams.n_factor, rho_cap=mparams.rho_cap)

        self.value = GGDLinear(self.D, self.all_head_size, k_bits_x=mparams.k_bits_x, k_bits_w=mparams.k_bits_w, N_factor=mparams.n_factor, rho_cap=mparams.rho_cap)
       
        self.output = GGDLinear(self.D, self.D, k_bits_x=mparams.k_bits_x, k_bits_w=mparams.k_bits_w, N_factor=mparams.n_factor, rho_cap=mparams.rho_cap)

        self.attn_dropout = nn.Dropout(mparams.attn_dropout)
        self.proj_dropout = nn.Dropout(mparams.attn_dropout)

        self.log_tau = nn.Parameter(torch.zeros(self.num_head, 1, 1))
    


    def forward(self, x, mask=None):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_head)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_head)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_head)

        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        attn_score = torch.matmul(q, k.transpose(-1, -2)) * (self.head_size ** -0.5)
        attn_score = attn_score * torch.exp(self.log_tau)


        

        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_score, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = rearrange(context, 'b h n d -> b n (h d)')
        output = self.output(context)         
        return self.proj_dropout(output)


class MLP(nn.Module):
    def __init__(self, mparams, hparams):
        super().__init__()
        self.D = mparams.inner_dim
        self.hidden_dim = 4* self.D
        self.net = nn.Sequential(
            GGDLinear(self.D, self.hidden_dim, k_bits_x=mparams.k_bits_x, k_bits_w=mparams.k_bits_w, N_factor=mparams.n_factor, rho_cap=mparams.rho_cap),
            OddGate(alpha=1.0),
            nn.Dropout(mparams.mlp_dropout),
            GGDLinear(self.hidden_dim, self.D, k_bits_x=mparams.k_bits_x, k_bits_w=mparams.k_bits_w, N_factor=mparams.n_factor, rho_cap=mparams.rho_cap),
            nn.Dropout(mparams.mlp_dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, mparams, hparams):
        super().__init__()
        self.norm1 = RMSNorm(mparams.inner_dim)
        self.attn = MHA(mparams=mparams, hparams=hparams)
        self.norm2 = RMSNorm(mparams.inner_dim)
        self.ffn = MLP(mparams=mparams, hparams=hparams)
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.attn(x) + residual

        residual = x
        x = self.norm2(x)
        x = self.ffn(x) + residual
        return x

class Transformer(nn.Module):
    def __init__(self, mparams, hparams):
        super().__init__()
        self.depth = mparams.transformer_layers
        self.layers = nn.ModuleList([
            EncoderBlock(mparams=mparams, hparams=hparams) for _ in range(self.depth)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ViT(nn.Module):
    def __init__(self, mparams, hparams, img_info):
        super().__init__()
        image_width = img_info.width
        patch_size = mparams.patch_size
        num_patches = (image_width//patch_size)**2
        self.pos_embed = nn.Parameter(torch.rand(1, num_patches+1, mparams.inner_dim))
        self.patch_embed = PatchEmbedding(mparams=mparams, hparams=hparams, img_info=img_info)
        self.transformer = Transformer(mparams=mparams, hparams=hparams)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(mparams.inner_dim),
            nn.Linear(mparams.inner_dim, hparams.out_classes)
        )
        self.embed_dropout = nn.Dropout(mparams.embed_dropout)
    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.embed_dropout(x)
        x = self.transformer(x)
        cls_token_ouput = x[:,0]
        return self.mlp_head(cls_token_ouput)

