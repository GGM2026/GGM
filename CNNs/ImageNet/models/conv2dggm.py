import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dGGM(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        N_factor: float = 2.5,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.eps = eps
        self.N_factor = N_factor

        assert in_channels % groups == 0
        assert out_channels % groups == 0

        cin_g = in_channels // groups
        K = cin_g * kernel_size * kernel_size
        self.N = max(500, int(K * N_factor))

        self.weight = nn.Parameter(torch.randn(out_channels, cin_g, kernel_size, kernel_size) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.scale = nn.Parameter(torch.ones(out_channels))

        self.register_buffer("G", torch.randn(groups, self.N, K))

    @torch.no_grad()
    def resample_G(self):
        self.G.copy_(torch.randn_like(self.G))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=(1, 2, 3), keepdim=True)
        g = self.groups
        k = self.kernel_size
        cin_g = self.in_channels // g
        cout_g = self.out_channels // g
        N = self.N

        Gx = self.G.to(device=x.device, dtype=x.dtype)
        G_conv = Gx.reshape(g * N, cin_g, k, k)
        z = F.conv2d(x, G_conv, stride=self.stride, padding=self.padding, groups=g)
        z_b = torch.where(
            z >= 0,
            torch.ones((), device=z.device, dtype=z.dtype),
            -torch.ones((), device=z.device, dtype=z.dtype),
        )

        W_flat = self.weight.reshape(g, cout_g, cin_g * k * k)
        Gw = self.G.to(device=x.device, dtype=W_flat.dtype)
        W_proj = torch.bmm(Gw, W_flat.transpose(1, 2))
        W_b = torch.where(
            W_proj >= 0,
            torch.ones((), device=W_proj.device, dtype=W_proj.dtype),
            -torch.ones((), device=W_proj.device, dtype=W_proj.dtype),
        ).transpose(1, 2).reshape(self.out_channels, N, 1, 1)

        y_bin = F.conv2d(z_b, W_b.to(z_b.dtype), groups=g) / float(N)

        x32 = x.float()
        W32 = self.weight.float()

        num = F.conv2d(x32, W32, stride=self.stride, padding=self.padding, groups=g)

        ones = torch.ones(g, cin_g, k, k, device=x.device, dtype=x32.dtype)
        p2 = F.conv2d(x32.square(), ones, stride=self.stride, padding=self.padding, groups=g)

        inv_pn = torch.rsqrt(p2 + self.eps).repeat_interleave(cout_g, dim=1)
        inv_wn = torch.rsqrt(W32.square().sum(dim=(1, 2, 3)) + self.eps).view(1, self.out_channels, 1, 1)

        corr = (num * inv_pn * inv_wn)
        y_surr = (2.0 / torch.pi) * torch.asin(corr).to(dtype=y_bin.dtype)

        y = y_bin.detach() + (y_surr - y_surr.detach())
        y = y * self.scale.view(1, -1, 1, 1)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size=({self.kernel_size}), "
            f"N={self.N}, "
            f"nu={self.N_factor}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"groups={self.groups}, "
            f"bias={self.bias is not None}, "
            f"G_shape={tuple(self.G.shape)}"
        )