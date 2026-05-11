import torch
import torch.nn as nn

class LinearGGM(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        N_factor: float = 1.0,
        bias: bool = False,
        eps: float = 1e-5,
        std: float = 0.02,
        x_norm: bool = False,
        w_norm: bool = False,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        self.weight = nn.Parameter(torch.randn(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None

        self.N_factor = float(N_factor)
        self.N = int(self.N_factor * self.in_features)
        self.register_buffer("G", torch.randn(self.N, self.in_features))
        self.eps = float(eps)

        self.x_norm = bool(x_norm)
        self.w_norm = bool(w_norm)

        nn.init.trunc_normal_(self.weight, std=std)

    @torch.no_grad()
    def resample_G(self):
        self.G.copy_(torch.randn_like(self.G))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, N={self.N}, "
            f"x_norm={self.x_norm}, w_norm={self.w_norm}"
        )

    def _centralize(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(torch.float32)
        mean = t.mean(dim=-1, keepdim=True)
        var = (t - mean).square().mean(dim=-1, keepdim=True)
        return (t - mean) / torch.sqrt(var + self.eps)

    def forward(self, x):
        x_norm = self._centralize(x) if self.x_norm else x.to(torch.float32)
        W_eff = self._centralize(self.weight) if self.w_norm else self.weight.to(torch.float32)

        W_b = (self.G @ W_eff.transpose(-1, -2)).sign()
        x_b = (x_norm @ self.G.transpose(-1, -2)).sign()
        y_bin = (x_b @ W_b) / self.N

        x32 = x_norm
        W32 = W_eff

        xnorm = (x32.square().sum(dim=-1, keepdim=True) + self.eps).sqrt()
        Wnorm = (W32.square().sum(dim=-1, keepdim=True) + self.eps).sqrt()

        xhat = x32 / xnorm
        What = W32 / Wnorm

        s = xhat @ What.transpose(-1, -2)
        s = s.clamp(-1.0 + 1e-4, 1.0 - 1e-4)
        y_surr = (2.0 / torch.pi) * torch.asin(s)
        y_surr = y_surr.to(dtype=y_bin.dtype)

        y = y_bin.detach() + (y_surr - y_surr.detach())

        if self.bias is not None:
            y = y + self.bias
        return y