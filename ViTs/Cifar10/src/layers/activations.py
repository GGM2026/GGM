import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLU2(nn.Module):
    def forward(self, x):
        y = F.relu(x)
        return y * y
    
class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently 
    in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) 
    paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (
            1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
            )
        )
        
class OddGate(nn.Module):
    def __init__(self, alpha=1.0, learnable=False):
        super().__init__()
        if learnable:
            self.log_alpha = nn.Parameter(torch.log(torch.tensor(float(alpha))))
        else:
            self.register_buffer("log_alpha", torch.log(torch.tensor(float(alpha))))

    def forward(self, z):
        alpha = torch.exp(self.log_alpha)
        return z * torch.tanh(alpha * z)

