import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        orig_dtype = x.dtype          # remember incoming dtype (fp16 / fp32)

        x_fp32 = x.float()            # force FP32 for norm math
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        y = x_fp32 / rms * self.weight

        return y.to(orig_dtype)       # return to original dtype for AMP

