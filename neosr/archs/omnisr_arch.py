# Code adapted from: https://github.com/Francis0625/Omni-SR

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

from neosr.utils.registry import ARCH_REGISTRY
from neosr.utils.options import parse_options

# initialize options parsing
root_path = Path(__file__).parents[2]
opt, args = parse_options(root_path, is_train=True)
# set scale factor in network parameters
upsampling = opt['scale']
ws = opt['network_g'].get('window_size')


def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3,
                       bias=False):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = nn.Conv2d(in_channels,
                     out_channels * (upscale_factor**2),
                     kernel_size,
                     padding=1,
                     bias=bias)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    return nn.Sequential(*[conv, pixel_shuffle])


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(
            dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by 
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv=nn.Conv2d):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear',
                           align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class PreNormResidual(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class Conv_PreNormResidual(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class Gated_Conv_FeedForward(nn.Module):

    def __init__(self, dim, mult=1, bias=False, dropout=0.):
        super().__init__()

        hidden_features = int(dim * mult)

        self.project_in = nn.Conv2d(dim,
                                    hidden_features * 2,
                                    kernel_size=1,
                                    bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2,
                                hidden_features * 2,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=hidden_features * 2,
                                bias=bias)

        self.project_out = nn.Conv2d(hidden_features,
                                     dim,
                                     kernel_size=1,
                                     bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class SqueezeExcitation(nn.Module):

    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(Reduce('b c h w -> b c', 'mean'),
                                  nn.Linear(dim, hidden_dim, bias=False),
                                  nn.SiLU(),
                                  nn.Linear(hidden_dim, dim, bias=False),
                                  nn.Sigmoid(), Rearrange('b c -> b c 1 1'))

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):

    def __init__(self, fn, dropout=0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


class Dropsample(nn.Module):

    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1),
                                      device=device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)


def MBConv(dim_in,
           dim_out,
           *,
           downsample,
           expansion_rate=4,
           shrinkage_rate=0.25,
           dropout=0.):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        # nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim,
                  hidden_dim,
                  3,
                  stride=stride,
                  padding=1,
                  groups=hidden_dim),
        # nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        # nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)

    return net


class Attention(nn.Module):

    def __init__(self, dim, dim_head=32, dropout=0., window_size=7, with_pe=True):
        super().__init__()
        assert (dim % dim_head
                ) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head**-0.5
        self.with_pe = with_pe

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))

        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False),
                                    nn.Dropout(dropout))

        # relative positional bias
        if self.with_pe:
            self.rel_pos_bias = nn.Embedding((2 * window_size - 1)**2,
                                             self.heads)

            pos = torch.arange(window_size)
            grid = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
            grid = rearrange(grid, 'c i j -> (i j) c')
            rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(
                grid, 'j ... -> 1 j ...')
            rel_pos += window_size - 1
            rel_pos_indices = (rel_pos *
                               torch.tensor([2 * window_size - 1, 1])).sum(
                                   dim=-1)

            self.register_buffer('rel_pos_indices',
                                 rel_pos_indices,
                                 persistent=False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # flatten
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h=h),
                      (q, k, v))

        # scale
        q = q * self.scale

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        if self.with_pe:
            bias = self.rel_pos_bias(self.rel_pos_indices)
            sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention
        attn = self.attend(sim)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out,
                        'b h (w1 w2) d -> b w1 w2 (h d)',
                        w1=window_height,
                        w2=window_width)

        # combine heads out
        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x=height, y=width)


class Channel_Attention(nn.Module):

    def __init__(self, dim, heads, bias=False, dropout=0., window_size=7):
        super(Channel_Attention, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3,
                                    dim * 3,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1)

        q, k, v = map(
            lambda t: rearrange(
                t,
                'b (head d) (h ph) (w pw) -> b (h w) head d (ph pw)',
                ph=self.ps,
                pw=self.ps,
                head=self.heads), qkv)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out,
                        'b (h w) head d (ph pw) -> b (head d) (h ph) (w pw)',
                        h=h // self.ps,
                        w=w // self.ps,
                        ph=self.ps,
                        pw=self.ps,
                        head=self.heads)

        out = self.project_out(out)

        return out


class Channel_Attention_grid(nn.Module):

    def __init__(self, dim, heads, bias=False, dropout=0., window_size=7):
        super(Channel_Attention_grid, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3,
                                    dim * 3,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1)

        q, k, v = map(
            lambda t: rearrange(
                t,
                'b (head d) (h ph) (w pw) -> b (ph pw) head d (h w)',
                ph=self.ps,
                pw=self.ps,
                head=self.heads), qkv)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out,
                        'b (ph pw) head d (h w) -> b (head d) (h ph) (w pw)',
                        h=h // self.ps,
                        w=w // self.ps,
                        ph=self.ps,
                        pw=self.ps,
                        head=self.heads)

        out = self.project_out(out)

        return out


class OSA_Block(nn.Module):

    def __init__(self,
                 channel_num=64,
                 bias=True,
                 ffn_bias=True,
                 window_size=8,
                 with_pe=False,
                 dropout=0.0):
        super(OSA_Block, self).__init__()

        w = window_size

        self.layer = nn.Sequential(
            MBConv(channel_num,
                   channel_num,
                   downsample=False,
                   expansion_rate=1,
                   shrinkage_rate=0.25),

            # block-like attention
            Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w,
                      w2=w),
            PreNormResidual(
                channel_num,
                Attention(dim=channel_num,
                          dim_head=channel_num // 4,
                          dropout=dropout,
                          window_size=window_size,
                          with_pe=with_pe)),
            Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
            Conv_PreNormResidual(
                channel_num,
                Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),

            # channel-like attention
            Conv_PreNormResidual(
                channel_num,
                Channel_Attention(dim=channel_num,
                                  heads=4,
                                  dropout=dropout,
                                  window_size=window_size)),
            Conv_PreNormResidual(
                channel_num,
                Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),

            # grid-like attention
            Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w,
                      w2=w),
            PreNormResidual(
                channel_num,
                Attention(dim=channel_num,
                          dim_head=channel_num // 4,
                          dropout=dropout,
                          window_size=window_size,
                          with_pe=with_pe)),
            Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
            Conv_PreNormResidual(
                channel_num,
                Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),

            # channel-like attention
            Conv_PreNormResidual(
                channel_num,
                Channel_Attention_grid(dim=channel_num,
                                       heads=4,
                                       dropout=dropout,
                                       window_size=window_size)),
            Conv_PreNormResidual(
                channel_num,
                Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class OSAG(nn.Module):

    def __init__(self,
                 channel_num=64,
                 bias=True,
                 block_num=4,
                 ffn_bias=True,
                 pe=True,
                 window_size=ws,
                 **kwargs):
        super(OSAG, self).__init__()

        group_list = []
        for _ in range(block_num):
            temp_res = OSA_Block(channel_num,
                                 bias,
                                 ffn_bias=ffn_bias,
                                 window_size=window_size,
                                 with_pe=pe)
            group_list.append(temp_res)
        group_list.append(
            nn.Conv2d(channel_num, channel_num, 1, 1, 0, bias=bias))
        self.residual_layer = nn.Sequential(*group_list)
        esa_channel = max(channel_num // 4, 16)
        self.esa = ESA(esa_channel, channel_num)

    def forward(self, x):
        out = self.residual_layer(x)
        out = out + x
        return self.esa(out)


@ARCH_REGISTRY.register()
class omnisr(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 res_num=5,
                 block_num=1,
                 bias=True,
                 pe=True,
                 ffn_bias=True,
                 upsampling=upsampling,
                 window_size=ws,
                 **kwargs):

        super(omnisr, self).__init__()

        self.window_size = window_size

        up_scale = upsampling
        self.up_scale = up_scale
        residual_layer = []

        for _ in range(res_num):
            temp_res = OSAG(channel_num=num_feat, **kwargs)
            residual_layer.append(temp_res)
        self.residual_layer = nn.Sequential(*residual_layer)
        self.input = nn.Conv2d(in_channels=num_in_ch,
                               out_channels=num_feat,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=bias)
        self.output = nn.Conv2d(in_channels=num_feat,
                                out_channels=num_feat,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=bias)
        self.up = pixelshuffle_block(num_feat, num_out_ch, up_scale, bias=bias)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        # import pdb; pdb.set_trace()
        mod_pad_h = (self.window_size -
                     h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size -
                     w % self.window_size) % self.window_size
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        residual = self.input(x)
        out = self.residual_layer(residual)

        # origin
        out = torch.add(self.output(out), residual)
        out = self.up(out)

        out = out[:, :, :H * self.up_scale, :W * self.up_scale]
        return out
