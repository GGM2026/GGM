import torch
import torch.nn as nn

from .layer_utils import make_G_from_seed

class LinearGGM(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        ratio: float = 1.0,
        bias: bool = False,
        eps: float = 1e-5,
        centralize_x: bool = False,
        centralize_w: bool = False,
        G_seed: int = 0,
        gain_init: float = 1.0, 
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
    
        self.weight = nn.Parameter(torch.randn(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
    
        self.ratio = float(ratio)
        self.N = int(self.ratio * self.in_features)
        self.G_seed = int(G_seed)
    
        G = make_G_from_seed(
            seed=self.G_seed,
            N=self.N,
            d=self.in_features,
            device=torch.device("cpu"),
        )
        self.register_buffer("G", G)
    
        self.eps = float(eps)
        self.centralize_x = bool(centralize_x)
        self.centralize_w = bool(centralize_w)
        
    
    @torch.no_grad()
    def resample_G(self):
        self.G.copy_(torch.randn_like(self.G))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, N={self.N}, "
            f"centralize_x={self.centralize_x}, centralize_w={self.centralize_w}"
        )

    def _centralize(self, t: torch.Tensor) -> torch.Tensor:
        t = t - t.mean(dim=-1, keepdim=True) 
        return t

    def forward(self, x):
        x_eff = self._centralize(x) if self.centralize_x else x.to(torch.float32)
        W_eff = self._centralize(self.weight) if self.centralize_w else self.weight.to(torch.float32)

        W_b = (self.G @ W_eff.transpose(-1, -2)).sign()
        x_b = (x_eff @ self.G.transpose(-1, -2)).sign()
        y_bin = (x_b @ W_b) / self.N

        x32 = x_eff
        W32 = W_eff

        xnorm = (x32.square().sum(dim=-1, keepdim=True) + self.eps).sqrt()
        Wnorm = (W32.square().sum(dim=-1, keepdim=True) + self.eps).sqrt()

        xhat = x32 / xnorm
        What = W32 / Wnorm

        s = xhat @ What.transpose(-1, -2)
        y_surr = (2.0 / torch.pi) * torch.asin(s)
        y_surr = y_surr.to(dtype=y_bin.dtype)

        y = y_bin.detach() + (y_surr - y_surr.detach())

        if self.bias is not None:
            y = y + self.bias

        return y