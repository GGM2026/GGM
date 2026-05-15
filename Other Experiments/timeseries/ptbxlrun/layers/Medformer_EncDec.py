import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv1dggm import Conv1dGGM
from .xnor_layers import Conv1dXNORNet
from .dorefa_layers import Conv1dDoReFa
from .rbnn_layers import Conv1dRBNN
from .irnet_layers import Conv1dIRNet
from .adabin_layers import Conv1dAdaBin, Maxout


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        d_ff,
        dropout,
        activation="relu",
        use_ggm: bool = False,
        use_xnor: bool = False,
        use_dorefa: bool = False,
        use_rbnn: bool = False,
        use_irnet: bool = False,
        use_adabin: bool = False,
    ):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        if sum([use_ggm, use_xnor, use_dorefa, use_adabin]) > 1:
            raise ValueError(
                "Only one of use_ggm, use_xnor, use_dorefa, use_rbnn, use_irnet can be True"
            )

        if use_ggm:
            self.conv1 = Conv1dGGM(
                in_channels=d_model,
                out_channels=d_ff,
                kernel_size=1,
                N_factor=3.0,
            )
            self.conv2 = Conv1dGGM(
                in_channels=d_ff,
                out_channels=d_model,
                kernel_size=1,
                N_factor=3.0,
            )

        elif use_xnor:
            self.conv1 = Conv1dXNORNet(
                in_channels=d_model,
                out_channels=d_ff,
                kernel_size=1,
                bias=True,
                binary_input=True,
                binary_weight=True,
            )
            self.conv2 = Conv1dXNORNet(
                in_channels=d_ff,
                out_channels=d_model,
                kernel_size=1,
                bias=True,
                binary_input=True,
                binary_weight=True,
            )

        elif use_dorefa:
            self.conv1 = Conv1dDoReFa(
                in_channels=d_model,
                out_channels=d_ff,
                kernel_size=1,
                bias=True,
                weight_bits=1,
                act_bits=1,
                quantize_input=True,
            )
            self.conv2 = Conv1dDoReFa(
                in_channels=d_ff,
                out_channels=d_model,
                kernel_size=1,
                bias=True,
                weight_bits=1,
                act_bits=1,
                quantize_input=True,
            )

        elif use_adabin:
            self.conv1 = Conv1dAdaBin(
                in_channels=d_model,
                out_channels=d_ff,
                kernel_size=1,
                bias=True,
            )
            self.conv2 = Conv1dAdaBin(
                in_channels=d_ff,
                out_channels=d_model,
                kernel_size=1,
                bias=True,
            )

        else:
            self.conv1 = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_ff,
                kernel_size=1,
            )
            self.conv2 = nn.Conv1d(
                in_channels=d_ff,
                out_channels=d_model,
                kernel_size=1,
            )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if use_adabin:
            self.activation = Maxout(d_ff)
        else:
            self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = [_x + self.dropout(_nx) for _x, _nx in zip(x, new_x)]

        y = x = [self.norm1(_x) for _x in x]
        y = [self.dropout(self.activation(self.conv1(_y.transpose(-1, 1)))) for _y in y]
        y = [self.dropout(self.conv2(_y).transpose(-1, 1)) for _y in y]

        return [self.norm2(_x + _y) for _x, _y in zip(x, y)], attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x is a list of tensors:
        # [[B, L1, D], [B, L2, D], ...]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        # concatenate all outputs along token dimension
        x = torch.cat(x, dim=1)  # (B, patch_num_1 + patch_num_2 + ..., D)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns