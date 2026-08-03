import torch
import torch.nn as nn
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import math
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import random

from .linearggm import LinearGGM

class BinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input


class ZMeanBinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        out[out==-1] = 0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input


class SymQuantizer(torch.autograd.Function):
    """
        uniform quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            if input.ndimension() <= 3:
                max_input = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_input = torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        s = (2 ** (num_bits - 1) - 1) / max_input
        output = torch.round(input * s).div(s)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class AsymQuantizer(torch.autograd.Function):
    """
        min-max quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            alpha = (input.max() - input.min()).detach()
            beta = input.min().detach()
        else:
            if input.ndimension() <= 3:
                alpha = (input.max(dim=-1, keepdim=True)[0] - input.min(dim=-1, keepdim=True)[0]).expand_as(input).detach()
                beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                tmp = input.view(input.shape[0], input.shape[1], -1)
                alpha = (tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1) - \
                            tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)).expand_as(input).detach()
                beta = tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        input_normalized = (input - beta) / (alpha + 1e-8)
        s = (2**num_bits - 1)
        quant_input = torch.round(input_normalized * s).div(s)
        output = quant_input * (alpha + 1e-8) + beta

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class TwnQuantizer(torch.autograd.Function):
    """Ternary Weight Networks (TWN)
    Ref: https://arxiv.org/abs/1605.04711
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param input: tensor to be ternarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            m = input.norm(p=1).div(input.nelement())
            thres = 0.7 * m
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = (mask * input).abs().sum() / mask.sum()
            result = alpha * pos - alpha * neg
        else: # row-wise only for embed / weight
            n = input[0].nelement()
            m = input.data.norm(p=1, dim=1).div(n)
            thres = (0.7 * m).view(-1, 1).expand_as(input)
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = ((mask * input).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
            result = alpha * pos - alpha * neg

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class QuantizeLinear(nn.Linear):
    def __init__(self,  in_features, out_features, bias=True, config=None, type=None):
        super().__init__(in_features, out_features,bias=bias)
        self.use_ggm = getattr(config, 'use_ggm', False)
        if self.use_ggm:
            # New clean LinearGGM API.
            # Backward-compatible config aliases:
            #   ggm_ratio -> N_factor
            #   input_bits -> k_bits_x
            #   weight_bits -> k_bits_w
            ggm_input_bits = getattr(config, "ggm_input_bits", getattr(config, "input_bits", 1))
            ggm_weight_bits = getattr(config, "ggm_weight_bits", getattr(config, "weight_bits", 1))

            ggm_kwargs = dict(
                in_features=in_features,
                out_features=out_features,
                k_bits_x=ggm_input_bits,
                k_bits_w=ggm_weight_bits,
                rho_eps=getattr(config, "ggm_rho_eps", 0.0),
                N_factor=getattr(config, "ggm_N_factor", getattr(config, "ggm_ratio", 1.0)),
                gain_init=getattr(config, "ggm_gain_init", 1.0),
                bias=bias,
                rho_cap=getattr(config, "ggm_rho_cap", 0.995),
                soft_rho=getattr(config, "ggm_soft_rho", False),
                table_grid_size=getattr(config, "ggm_table_grid_size", 2048),
                table_bits_min=getattr(config, "ggm_table_bits_min", 2),
                table_bits_max=getattr(config, "ggm_table_bits_max", 5),
                table_s0_x=getattr(config, "ggm_table_s0_x", 1.0),
                table_s0_w=getattr(config, "ggm_table_s0_w", 1.0),
                use_centering=getattr(config, "ggm_use_centering", True),
                use_std_norm=getattr(config, "ggm_use_std_norm", False),
                scale_policy=getattr(config, "ggm_scale_policy", "learnable_mean"),
                fixed_x_scale=getattr(config, "ggm_fixed_x_scale", 1.0),
                fixed_w_scale=getattr(config, "ggm_fixed_w_scale", 1.0),
            )

            ggm_seed = getattr(config, "ggm_seed", None)
            if ggm_seed is None:
                self.ggm = LinearGGM(**ggm_kwargs)
            else:
                # New LinearGGM samples its own G_seed internally. Fork RNG so
                # ggm_seed is reproducible without perturbing the caller RNG.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(ggm_seed))
                    self.ggm = LinearGGM(**ggm_kwargs)

            # Share the QuantizeLinear parameters with LinearGGM so checkpoints
            # and optimizer parameter groups keep seeing self.weight/self.bias.
            self.ggm.weight = self.weight
            self.ggm.bias = self.bias if bias else None

        else:
            self.quantize_act = config.quantize_act
            self.weight_bits = config.weight_bits
            self.quantize_act = config.quantize_act
            if self.weight_bits == 2:
                self.weight_quantizer = TwnQuantizer
            elif self.weight_bits == 1:
                self.weight_quantizer = BinaryQuantizer
            else:
                self.weight_quantizer = SymQuantizer
            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
            
            if self.quantize_act:
                self.input_bits = config.input_bits
                if self.input_bits == 1:
                    self.act_quantizer = BinaryQuantizer
                elif self.input_bits == 2:
                    self.act_quantizer = TwnQuantizer
                else:
                    self.act_quantizer = SymQuantizer
                self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
            self.register_parameter('scale', Parameter(torch.Tensor([0.0]).squeeze()))
 
    def reset_scale(self, input):
        if self.use_ggm:
            return
        bw = self.weight
        ba = input
        self.scale = Parameter((ba.norm() / torch.sign(ba).norm()).float().to(ba.device))

    def forward(self, input, type=None):
        if self.use_ggm:
            return self.ggm(input)
        if self.weight_bits == 1:
            scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
            scaling_factor = scaling_factor.detach()
            real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
            binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
            cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
            weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        else:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)

        if self.input_bits == 1:
            binary_input_no_grad = torch.sign(input)
            cliped_input = torch.clamp(input, -1.0, 1.0)
            ba = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input
        else:
            ba = self.act_quantizer.apply(input, self.act_clip_val, self.input_bits, True)
        
        out = nn.functional.linear(ba, weight)
        
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out) 

        return out


class QuantizeEmbedding(nn.Embedding):
    def __init__(self,  *kargs,padding_idx=None, config=None, type=None):
        super(QuantizeEmbedding, self).__init__(*kargs, padding_idx = padding_idx)
        self.weight_bits = config.weight_bits
        self.layerwise = False
        if self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
        elif self.weight_bits == 1:
            self.weight_quantizer = BinaryQuantizer
        else:
            self.weight_quantizer = SymQuantizer
        self.init = True
        self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))

    def forward(self, input, type=None):
        if self.weight_bits == 1:
            scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
            scaling_factor = scaling_factor.detach()
            real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
            binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
            cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
            weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        else:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, self.layerwise)
        out = nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        return out
