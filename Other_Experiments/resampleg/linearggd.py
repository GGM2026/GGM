import torch
import torch.nn as nn

class LinearGGD(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        N_scale: float = 1.0,
        bias: bool = True,
        eps: float = 1e-5,
        std: float = 0.02,
        g_seed = None,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.randn(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None

        self.N_scale = float(N_scale)
        self.N = int(self.N_scale * self.in_features)
        self.eps = float(eps)
        self.g_seed = g_seed

        self.register_buffer("G", self._make_G())

        nn.init.trunc_normal_(self.weight, std=std)

    def _make_G(self):
        if self.g_seed is None:
            return torch.randn(self.N, self.in_features)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.g_seed)
        return torch.randn(self.N, self.in_features, generator=gen)

    @torch.no_grad()
    def resample_G(self, seed=None):
        if seed is not None:
            self.g_seed = seed
        self.G.copy_(self._make_G())

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, N={self.N}, G_shape={tuple(self.G.shape)}, "
            f"g_seed={self.g_seed}"
        )

    def forward(self, x):
        G = self.G.to(device=x.device, dtype=x.dtype)

        W_b = (G @ self.weight.transpose(-1, -2).to(G.dtype)).sign()
        x_b = (x @ G.transpose(-1, -2)).sign()
        y_bin = (x_b @ W_b) / self.N

        x32 = x.to(torch.float32)
        W32 = self.weight.to(torch.float32)

        xnorm = (x32.square() + self.eps).sum(dim=-1, keepdim=True).sqrt()
        Wnorm = (W32.square() + self.eps).sum(dim=-1, keepdim=True).sqrt()

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