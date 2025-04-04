import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Optional, Dict, Union
from einops import rearrange, reduce
from collections import OrderedDict

from ..backbone.UniRepLKNet import get_bn, get_conv2d, NCHWtoNHWC, GRNwithNHWC, SEBlock, NHWCtoNCHW, fuse_bn, merge_dilated_into_large_kernel
from ..backbone.rmt import RetBlock, RelPos2d
from ..modules.conv import Conv, DWConv, DSConv, RepConv, GhostConv, autopad, LightConv, ConvTranspose
from ..modules.block import get_activation, ConvNormLayer, BasicBlock, BottleNeck, RepC3, C3, C2f, Bottleneck
from .attention import *
from .ops_dcnv3.modules import DCNv3
from .transformer import LocalWindowAttention
from .dynamic_snake_conv import DySnakeConv
from .RFAConv import RFAConv, RFCAConv, RFCBAMConv
from .rep_block import *
from .shiftwise_conv import ReparamLargeKernelConv
from .mamba_vss import VSSBlock
from .orepa import OREPA
from .fadc import AdaptiveDilatedConv
from .hcfnet import PPA
from .deconv import DEConv
from .SMPConv import SMPConv
from .kan_convs import FastKANConv2DLayer, KANConv2DLayer, KALNConv2DLayer, KACNConv2DLayer, KAGNConv2DLayer
from .wtconv2d import WTConv2d
from .camixer import CAMixer
from .tsdn import DTAB, LayerNorm
from ultralytics.utils.torch_utils import fuse_conv_and_bn, make_divisible

from timm.layers import trunc_normal_
from timm.layers import DropPath

__all__ = ['SobelConv', 'MutilScaleEdgeInfoGenetator', 'ConvEdgeFusion',
           'GSA', 'RSA', 'C2f_FDT','WUM','HaarWavelet', 'WUFF','Fusion'
           'WDM'

           ]


########################################Edg start ########################################

class SobelConv(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()

        sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_kernel_y = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        sobel_kernel_x = torch.tensor(sobel.T, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)

        self.sobel_kernel_x_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.sobel_kernel_y_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)

        self.sobel_kernel_x_conv3d.weight.data = sobel_kernel_x.clone()
        self.sobel_kernel_y_conv3d.weight.data = sobel_kernel_y.clone()

        self.sobel_kernel_x_conv3d.requires_grad = False
        self.sobel_kernel_y_conv3d.requires_grad = False

    def forward(self, x):
        return (self.sobel_kernel_x_conv3d(x[:, :, None, :, :]) + self.sobel_kernel_y_conv3d(x[:, :, None, :, :]))[:, :,
               0]


class MutilScaleEdgeInfoGenetator(nn.Module):
    def __init__(self, inc, oucs) -> None:
        super().__init__()

        self.sc = SobelConv(inc)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1x1s = nn.ModuleList(Conv(inc, ouc, 1) for ouc in oucs)

    def forward(self, x):
        outputs = [self.sc(x)]
        outputs.extend(self.maxpool(outputs[-1]) for _ in self.conv_1x1s)
        outputs = outputs[1:]
        for i in range(len(self.conv_1x1s)):
            outputs[i] = self.conv_1x1s[i](outputs[i])
        return outputs


class ConvEdgeFusion(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()

        self.conv_channel_fusion = Conv(sum(inc), ouc // 2, k=1)
        self.conv_3x3_feature_extract = Conv(ouc // 2, ouc // 2, 3)
        self.conv_1x1 = Conv(ouc // 2, ouc, 1)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = self.conv_1x1(self.conv_3x3_feature_extract(self.conv_channel_fusion(x)))
        return x

######################################## Edge end ########################################

######################################## wuff ########################################

class GSA(nn.Module):
    def __init__(self, channels, num_heads=8, bias=False):
        super(GSA, self).__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.act = nn.ReLU()

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, stride=1, padding=1, groups=channels * 3,
                                    bias=bias)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x, prev_atns=None):
        b, c, h, w = x.shape
        if prev_atns is None:
            qkv = self.qkv_dwconv(self.qkv(x))
            q, k, v = qkv.chunk(3, dim=1)
            q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = self.act(attn)
            out = (attn @ v)
            y = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
            y = rearrange(y, 'b (head c) h w -> b (c head) h w', head=self.num_heads, h=h, w=w)
            y = self.project_out(y)
            return y, attn
        else:
            attn = prev_atns
            v = rearrange(x, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            out = (attn @ v)
            y = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
            y = rearrange(y, 'b (head c) h w -> b (c head) h w', head=self.num_heads, h=h, w=w)
            y = self.project_out(y)
            return y


class RSA(nn.Module):
    def __init__(self, channels, num_heads, shifts=1, window_sizes=[4, 8, 12], bias=False):
        super(RSA, self).__init__()
        self.channels = channels
        self.shifts = shifts
        self.window_sizes = window_sizes

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.act = nn.ReLU()

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, stride=1, padding=1, groups=channels * 3,
                                    bias=bias)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x, prev_atns=None):
        b, c, h, w = x.shape
        if prev_atns is None:
            wsize = self.window_sizes
            x_ = x
            if self.shifts > 0:
                x_ = torch.roll(x_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
            qkv = self.qkv_dwconv(self.qkv(x_))
            q, k, v = qkv.chunk(3, dim=1)
            q = rearrange(q, 'b c (h dh) (w dw) -> b (h w) (dh dw) c', dh=wsize, dw=wsize)
            k = rearrange(k, 'b c (h dh) (w dw) -> b (h w) (dh dw) c', dh=wsize, dw=wsize)
            v = rearrange(v, 'b c (h dh) (w dw) -> b (h w) (dh dw) c', dh=wsize, dw=wsize)

            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

            attn = (q.transpose(-2, -1) @ k) * self.temperature  # b (h w) (dh dw) (dh dw)
            attn = self.act(attn)
            out = (v @ attn)
            out = rearrange(out, 'b (h w) (dh dw) c-> b (c) (h dh) (w dw)', h=h // wsize, w=w // wsize, dh=wsize,
                            dw=wsize)
            if self.shifts > 0:
                out = torch.roll(out, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
            y = self.project_out(out)
            return y, attn
        else:
            wsize = self.window_sizes
            if self.shifts > 0:
                x = torch.roll(x, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
            atn = prev_atns
            v = rearrange(x, 'b (c) (h dh) (w dw) -> b (h w) (dh dw) c', dh=wsize, dw=wsize)
            y_ = (v @ atn)
            y_ = rearrange(y_, 'b (h w) (dh dw) c-> b (c) (h dh) (w dw)', h=h // wsize, w=w // wsize, dh=wsize,
                           dw=wsize)
            if self.shifts > 0:
                y_ = torch.roll(y_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
            y = self.project_out(y_)
            return y


class WUM(nn.Module):
    def __init__(self, inp_channels, num_heads=4, window_sizes=4, shifts=0, shared_depth=1, ffn_expansion_factor=2.66):
        super(FDT, self).__init__()
        self.shared_depth = shared_depth

        modules_ffd = {}
        modules_att = {}
        modules_norm = {}
        for i in range(shared_depth):
            modules_ffd['ffd{}'.format(i)] = FeedForward(inp_channels, ffn_expansion_factor, bias=False)
            modules_att['att_{}'.format(i)] = RSA(channels=inp_channels, num_heads=num_heads, shifts=shifts,
                                                  window_sizes=window_sizes)
            modules_norm['norm_{}'.format(i)] = LayerNorm(inp_channels, 'WithBias')
            modules_norm['norm_{}'.format(i + 2)] = LayerNorm(inp_channels, 'WithBias')
        self.modules_ffd = nn.ModuleDict(modules_ffd)
        self.modules_att = nn.ModuleDict(modules_att)
        self.modules_norm = nn.ModuleDict(modules_norm)

        modulec_ffd = {}
        modulec_att = {}
        modulec_norm = {}
        for i in range(shared_depth):
            modulec_ffd['ffd{}'.format(i)] = FeedForward(inp_channels, ffn_expansion_factor, bias=False)
            modulec_att['att_{}'.format(i)] = GSA(channels=inp_channels, num_heads=num_heads)
            modulec_norm['norm_{}'.format(i)] = LayerNorm(inp_channels, 'WithBias')
            modulec_norm['norm_{}'.format(i + 2)] = LayerNorm(inp_channels, 'WithBias')
        self.modulec_ffd = nn.ModuleDict(modulec_ffd)
        self.modulec_att = nn.ModuleDict(modulec_att)
        self.modulec_norm = nn.ModuleDict(modulec_norm)

    def forward(self, x):
        atn = None
        B, C, H, W = x.size()
        for i in range(self.shared_depth):
            if i == 0:  ## only calculate attention for the 1-st module
                x_, atn = self.modules_att['att_{}'.format(i)](self.modules_norm['norm_{}'.format(i)](x), None)
                x = self.modules_ffd['ffd{}'.format(i)](self.modules_norm['norm_{}'.format(i + 2)](x_ + x)) + x_
            else:
                x_ = self.modules_att['att_{}'.format(i)](self.modules_norm['norm_{}'.format(i)](x), atn)
                x = self.modules_ffd['ffd{}'.format(i)](self.modules_norm['norm_{}'.format(i + 2)](x_ + x)) + x_

        for i in range(self.shared_depth):
            if i == 0:  ## only calculate attention for the 1-st module
                x_, atn = self.modulec_att['att_{}'.format(i)](self.modulec_norm['norm_{}'.format(i)](x), None)
                x = self.modulec_ffd['ffd{}'.format(i)](self.modulec_norm['norm_{}'.format(i + 2)](x_ + x)) + x_
            else:
                x = self.modulec_att['att_{}'.format(i)](self.modulec_norm['norm_{}'.format(i)](x), atn)
                x = self.modulec_ffd['ffd{}'.format(i)](self.modulec_norm['norm_{}'.format(i + 2)](x_ + x)) + x_

        return x


class C2f_FDT(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(FDT(self.c) for _ in range(n))


class HaarWavelet(nn.Module):
    def __init__(self, in_channels, grad=False):
        super(HaarWavelet, self).__init__()
        self.in_channels = in_channels

        self.haar_weights = torch.ones(4, 1, 2, 2)
        # h
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1
        # v
        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1
        # d
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = grad

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.in_channels) / 4.0
            out = out.reshape([x.shape[0], self.in_channels, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            out = x.reshape([x.shape[0], 4, self.in_channels, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.in_channels)


class WUFF(nn.Module):
    def __init__(self, chn):
        super(WFU, self).__init__()
        dim_big, dim_small = chn
        self.dim = dim_big
        self.HaarWavelet = HaarWavelet(dim_big, grad=False)
        self.InverseHaarWavelet = HaarWavelet(dim_big, grad=False)
        self.RB = nn.Sequential(
            # nn.Conv2d(dim_big, dim_big, kernel_size=3, padding=1),
            # nn.ReLU(),
            Conv(dim_big, dim_big, 3),
            nn.Conv2d(dim_big, dim_big, kernel_size=3, padding=1),
        )

        self.channel_tranformation = nn.Sequential(
            # nn.Conv2d(dim_big+dim_small, dim_big+dim_small // 1, kernel_size=1, padding=0),
            # nn.ReLU(),
            Conv(dim_big + dim_small, dim_big + dim_small // 1, 1),
            nn.Conv2d(dim_big + dim_small // 1, dim_big * 3, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x_big, x_small = x
        haar = self.HaarWavelet(x_big, rev=False)
        a = haar.narrow(1, 0, self.dim)
        h = haar.narrow(1, self.dim, self.dim)
        v = haar.narrow(1, self.dim * 2, self.dim)
        d = haar.narrow(1, self.dim * 3, self.dim)

        hvd = self.RB(h + v + d)
        a_ = self.channel_tranformation(torch.cat([x_small, a], dim=1))
        out = self.InverseHaarWavelet(torch.cat([hvd, a_], dim=1), rev=True)
        return out

######################################## wuff ########################################

######################################## WDM start ########################################

class WDM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HWD, self).__init__()
        from pytorch_wavelets import DWTForward
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv = Conv(in_ch * 4, out_ch, 1, 1)

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv(x)

        return x

######################################## WDM end ########################################

######################################## WCFPN begin ########################################

class Fusion(nn.Module):
    def __init__(self, inc_list, fusion='bifpn') -> None:
        super().__init__()

        assert fusion in ['weight', 'adaptive', 'concat', 'bifpn', 'SDI']
        self.fusion = fusion

        if self.fusion == 'bifpn':
            self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
            self.relu = nn.ReLU()
            self.epsilon = 1e-4
        elif self.fusion == 'SDI':
            self.SDI = SDI(inc_list)
        else:
            self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list])

            if self.fusion == 'adaptive':
                self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)

    def forward(self, x):
        if self.fusion in ['weight', 'adaptive']:
            for i in range(len(x)):
                x[i] = self.fusion_conv[i](x[i])
        if self.fusion == 'weight':
            return torch.sum(torch.stack(x, dim=0), dim=0)
        elif self.fusion == 'adaptive':
            fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)
            x_weight = torch.split(fusion, [1] * len(x), dim=1)
            return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'concat':
            return torch.cat(x, dim=1)
        elif self.fusion == 'bifpn':
            fusion_weight = self.relu(self.fusion_weight.clone())
            fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0))
            return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'SDI':
            return self.SDI(x)

######################################## WCFPN end ########################################
