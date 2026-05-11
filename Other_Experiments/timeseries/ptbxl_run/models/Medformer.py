import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Medformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import MedformerLayer
from layers.Embed import ListPatchEmbedding
import numpy as np


class Model(nn.Module):
    """
    Medformer
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in
        self.single_channel = configs.single_channel

        self.use_ggd = getattr(configs, "use_ggd", False)
        self.use_xnor = getattr(configs, "use_xnor", False)
        self.use_dorefa = getattr(configs, "use_dorefa", False)
        self.use_adabin = getattr(configs, "use_adabin", False)

        # Embedding
        patch_len_list = list(map(int, configs.patch_len_list.split(",")))
        stride_list = patch_len_list
        seq_len = configs.seq_len
        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]
        augmentations = configs.augmentations.split(",")

        self.enc_embedding = ListPatchEmbedding(
            configs.enc_in,
            configs.d_model,
            patch_len_list,
            stride_list,
            configs.dropout,
            augmentations,
            configs.single_channel,
        )
        
        if sum([
            self.use_ggd,
            self.use_xnor,
            self.use_dorefa,
            self.use_adabin,
        ]) > 1:
            raise ValueError(
                "Only one of use_ggd, use_xnor, use_dorefa or use_adabin can be True"
            )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MedformerLayer(
                        len(patch_len_list),
                        configs.d_model,
                        configs.n_heads,
                        dropout=configs.dropout,
                        output_attention=configs.output_attention,
                        no_inter=configs.no_inter_attn,
                        use_ggd=self.use_ggd,
                        use_xnor=self.use_xnor,
                        use_dorefa=self.use_dorefa,
                        use_adabin=self.use_adabin
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    use_ggd=self.use_ggd,
                    use_xnor=self.use_xnor,
                    use_dorefa=self.use_dorefa,
                    use_adabin=self.use_adabin
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # Classification head
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)

            head_in_dim = (
                configs.d_model
                * sum(patch_num_list)
                * (1 if not self.single_channel else configs.enc_in)
            )

            self.projection = nn.Linear(head_in_dim, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        raise NotImplementedError

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        raise NotImplementedError

    def anomaly_detection(self, x_enc):
        raise NotImplementedError

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        if self.single_channel:
            enc_out = torch.reshape(enc_out, (-1, self.enc_in, *enc_out.shape[-2:]))

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]
        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None