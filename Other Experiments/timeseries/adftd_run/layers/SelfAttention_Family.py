# layers/SelfAttention_Family
import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from .linearggm import LinearGGM
from .xnor_layers import LinearXNORNet
from .dorefa_layers import LinearDoReFa
from .adabin_layers import LinearAdaBin
import math



class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale if self.scale is not None else 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(
            torch.softmax(scale * scores, dim=-1)
        )  # Scaled Dot-Product Attention
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None



class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        n_heads,
        d_keys=None,
        d_values=None,
        use_ggm: bool = False,
        use_xnor: bool = False,
        use_dorefa: bool = False,
        use_adabin: bool = False,
    ):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.n_heads = n_heads

        if sum([use_ggm, use_xnor, use_dorefa, use_adabin]) > 1:
            raise ValueError("Only one of use_ggm, use_xnor, use_dorefa or use_adabin can be True")

        if use_ggm:
            self.query_projection = LinearGGM(d_model, d_keys * n_heads, N_scale=5.0)
            self.key_projection = LinearGGM(d_model, d_keys * n_heads, N_scale=5.0)
            self.value_projection = LinearGGM(d_model, d_values * n_heads, N_scale=5.0)
            self.out_projection = LinearGGM(d_values * n_heads, d_model, N_scale=5.0)
        elif use_xnor:
            self.query_projection = LinearXNORNet(d_model, d_keys * n_heads)
            self.key_projection = LinearXNORNet(d_model, d_keys * n_heads)
            self.value_projection = LinearXNORNet(d_model, d_values * n_heads)
            self.out_projection = LinearXNORNet(d_values * n_heads, d_model)
        elif use_dorefa:
            self.query_projection = LinearDoReFa(d_model, d_keys * n_heads, weight_bits=1, act_bits=1)
            self.key_projection = LinearDoReFa(d_model, d_keys * n_heads, weight_bits=1, act_bits=1)
            self.value_projection = LinearDoReFa(d_model, d_values * n_heads, weight_bits=1, act_bits=1)
            self.out_projection = LinearDoReFa(d_values * n_heads, d_model, weight_bits=1, act_bits=1)
        elif use_adabin:
            self.query_projection = LinearAdaBin(d_model, d_keys * n_heads, bias=True)
            self.key_projection = LinearAdaBin(d_model, d_keys * n_heads, bias=True)
            self.value_projection = LinearAdaBin(d_model, d_values * n_heads, bias=True)
            self.out_projection = LinearAdaBin(d_values * n_heads, d_model, bias=True)
        else:
            self.query_projection = nn.Linear(d_model, d_keys * n_heads)
            self.key_projection = nn.Linear(d_model, d_keys * n_heads)
            self.value_projection = nn.Linear(d_model, d_values * n_heads)
            self.out_projection = nn.Linear(d_values * n_heads, d_model)
            
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class MedformerLayer(nn.Module):
    def __init__(
        self,
        num_blocks,
        d_model,
        n_heads,
        dropout=0.1,
        output_attention=False,
        no_inter=False,
        use_ggm: bool = False,
        use_xnor: bool = False,
        use_dorefa: bool = False,
        use_adabin: bool = False,
    ):
        super().__init__()
        self.use_ggm = use_ggm
        self.use_xnor = use_xnor
        self.use_dorefa = use_dorefa
        self.use_adabin = use_adabin
        self.head_dim = d_model // n_heads

        if self.use_ggm:
            self.scale = d_model * math.pi**2 / (4 * math.sqrt(self.head_dim))
        else:
            self.scale = 1 / math.sqrt(self.head_dim)

        self.intra_attentions = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(
                        False,
                        factor=1,
                        attention_dropout=dropout,
                        output_attention=output_attention,
                        scale=self.scale,
                    ),
                    d_model,
                    n_heads,
                    use_ggm=self.use_ggm,
                    use_xnor=self.use_xnor,
                    use_dorefa=self.use_dorefa,
                    use_adabin=self.use_adabin,
                )
                for _ in range(num_blocks)
            ]
        )

        if no_inter or num_blocks <= 1:
            self.inter_attention = None
        else:
            self.inter_attention = AttentionLayer(
                FullAttention(
                    False,
                    factor=1,
                    attention_dropout=dropout,
                    output_attention=output_attention,
                    scale=self.scale,
                ),
                d_model,
                n_heads,
                use_ggm=self.use_ggm,
                use_xnor=self.use_xnor,
                use_dorefa=self.use_dorefa,
                use_adabin=self.use_adabin,
            )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attn_mask = attn_mask or ([None] * len(x))
        # Intra attention
        x_intra = []
        attn_out = []
        for x_in, layer, mask in zip(x, self.intra_attentions, attn_mask):
            _x_out, _attn = layer(x_in, x_in, x_in, attn_mask=mask, tau=tau, delta=delta)
            x_intra.append(_x_out)  # (B, Li, D)
            attn_out.append(_attn)
        if self.inter_attention is not None:
            # Inter attention
            routers = torch.cat([x[:, -1:] for x in x_intra], dim=1)  # (B, N, D)
            x_inter, attn_inter = self.inter_attention(
                routers, routers, routers, attn_mask=None, tau=tau, delta=delta
            )
            x_out = [
                torch.cat([x[:, :-1], x_inter[:, i : i + 1]], dim=1)  # (B, Li, D)
                for i, x in enumerate(x_intra)
            ]
            attn_out += [attn_inter]
        else:
            x_out = x_intra
        return x_out, attn_out
