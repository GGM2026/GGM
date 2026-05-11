import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class single_layer_matrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, G, N):
        eps = 1e-5

        
        W_b = (G @ W.t()).sign()
        x_b = (x @ G.t()).sign()

        y_pred = (x_b @ W_b) / N
        
        ctx.save_for_backward(x, W)
        ctx.N = N
        ctx.eps = eps

        return y_pred

    @staticmethod
    def backward(ctx, dy):
        x, W = ctx.saved_tensors
        N = ctx.N
        eps = ctx.eps
        with torch.enable_grad():
            x_ = x.clone().detach().requires_grad_(True)
            W_ = W.clone().detach().requires_grad_(True)

            Wnorm = (W_.pow(2) + eps).sum(dim=-1, keepdim=True).sqrt()
            xnorm = (x_.pow(2) + eps).sum(dim=-1, keepdim=True).sqrt()

            W_normed = W_ / Wnorm
            x_normed = x_ / xnorm

            asu = (2 / (torch.pi)) * torch.asin(x_normed @ W_normed.t())
            x_grad, W_grad = torch.autograd.grad(asu, [x_, W_], grad_outputs=dy, retain_graph=False)

        return x_grad, W_grad, None, None

class LinearGGD(nn.Module):
    def __init__(self, in_features, out_features, N_scale: float = 1.0, bias: bool = False):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.randn(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None

        self.N_scale = float(N_scale)
        self.N = int(self.N_scale * self.in_features)

        G = torch.randn(self.N, self.in_features)
        self.register_buffer("G", G)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, hyperdim_scale={self.N_scale}")

    @torch.no_grad()
    def resample_G(self):
        self.G.copy_(torch.randn_like(self.G))

    def forward(self, x):
        output = single_layer_matrix.apply(x, self.weight, self.G, self.N)
        return output + self.bias if self.bias is not None else output




        
        