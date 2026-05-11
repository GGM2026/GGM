import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dGGD(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        N_scale=1.0,
        stride=1,
        padding=0,
        groups=1,
        bias=False,
        eps=1e-6,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.eps = eps

        assert in_channels % groups == 0
        assert out_channels % groups == 0

        cin_g = in_channels // groups
        k = kernel_size
        K = cin_g * k * k
        self.N = max(500, int(K * N_scale))

        self.weight = nn.Parameter(torch.randn(out_channels, cin_g, k, k) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.scale = nn.Parameter(torch.ones(out_channels))

        G = torch.sign(torch.randn(groups, self.N, K))
        self.register_buffer("G", G)

    def forward(self, x):
        g = self.groups
        k = self.kernel_size
        cin_g = self.in_channels // g
        cout_g = self.out_channels // g
        N = self.N

        # ----------------------------
        # binary forward
        # ----------------------------
        Gx = self.G.to(device=x.device, dtype=x.dtype)
        G_conv = Gx.reshape(g * N, cin_g, k, k)

        z = F.conv2d(
            x, G_conv,
            stride=self.stride,
            padding=self.padding,
            groups=g,
        )
        z_b = torch.where(z >= 0, torch.ones((), device=z.device, dtype=z.dtype),
                               -torch.ones((), device=z.device, dtype=z.dtype))

        W_flat = self.weight.reshape(g, cout_g, cin_g * k * k)
        Gw = self.G.to(device=x.device, dtype=W_flat.dtype)
        W_proj = torch.bmm(Gw, W_flat.transpose(1, 2))   # (g, N, cout_g)
        W_b = torch.where(
            W_proj >= 0,
            torch.ones((), device=W_proj.device, dtype=W_proj.dtype),
            -torch.ones((), device=W_proj.device, dtype=W_proj.dtype),
        )
        W_b = W_b.transpose(1, 2).reshape(self.out_channels, N, 1, 1)

        y_bin = F.conv2d(z_b, W_b.to(z_b.dtype), groups=g) / float(N)

        # ----------------------------
        # smooth surrogate forward
        # ----------------------------
        x32 = x.float()
        W32 = self.weight.float()

        num = F.conv2d(
            x32, W32,
            stride=self.stride,
            padding=self.padding,
            groups=g,
        )

        ones = torch.ones(
            g, cin_g, k, k,
            device=x.device,
            dtype=x32.dtype,
        )
        p2 = F.conv2d(
            x32.square(), ones,
            stride=self.stride,
            padding=self.padding,
            groups=g,
        )  # (B, g, outH, outW)

        inv_pn = torch.rsqrt(p2 + self.eps) # (B, g, outH, outW)
        inv_pn = inv_pn.repeat_interleave(cout_g, dim=1) # (B, Cout, outH, outW)

        inv_wn = torch.rsqrt(
            W32.square().sum(dim=(1, 2, 3)) + self.eps
        ).view(1, self.out_channels, 1, 1)

        corr = num * inv_pn * inv_wn
        corr = corr.clamp(-1.0 + 1e-4, 1.0 - 1e-4)

        y_surr = (2.0 / torch.pi) * torch.asin(corr)
        y_surr = y_surr.to(dtype=y_bin.dtype)

        # forward = binary, backward = surrogate
        y = y_bin.detach() + (y_surr - y_surr.detach())

        y = y * self.scale.view(1, -1, 1, 1)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y