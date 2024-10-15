from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from neosr.archs.arch_util import DropPath, net_opt
from neosr.utils.registry import ARCH_REGISTRY

upscale, __ = net_opt()


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super().__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x):
        return self.cab(x)


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv):
        super().__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(
            c3, (x.size(2), x.size(3)), mode="bilinear", align_corners=False
        )
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class ConvBN(torch.nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        with_bn=True,
    ):
        super().__init__()
        self.add_module(
            "conv",
            torch.nn.Conv2d(
                in_planes, out_planes, kernel_size, stride, padding, dilation, groups
            ),
        )
        if with_bn:
            self.add_module("bn", torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.0):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.ln = nn.LayerNorm(dim)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f3 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=False)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.esa = ESA(16, 156, nn.Conv2d)
        self.cab = CAB(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x_orig = x
        x = self.dwconv(x)

        _N, _C, _H, _W = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()  # Change shape to (N, H, W, C)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # Change back to (N, C, H, W)

        x1, x2, x3 = self.f1(x), self.f2(x), self.f3(x)
        x = self.act(x1) * x2
        x3 = self.esa(x3)
        x = self.dwconv2(self.g(x + x3))
        x = x_orig + self.drop_path(x)
        return self.cab(x)


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels, out_channels, kernel_size, bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def sequential(*args):
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.

    Parameters
    ----------
    args: Definition of Modules in order.
    -------

    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            msg = "sequential does not support OrderedDict input."
            raise NotImplementedError(msg)
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = conv_layer(in_channels, out_channels * (upscale_factor**2), kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


@ARCH_REGISTRY.register()
class hasn(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=3, feature_channels=52, upscale=upscale
    ):
        super().__init__()

        self.conv_1 = conv_layer(in_channels, feature_channels, kernel_size=3)

        self.block_1 = Block(feature_channels)
        self.block_2 = Block(feature_channels)
        self.block_3 = Block(feature_channels)
        self.block_4 = Block(feature_channels)
        self.block_5 = Block(feature_channels)
        self.block_6 = Block(feature_channels)

        self.conv_2 = conv_layer(feature_channels, feature_channels, kernel_size=3)

        self.upsampler = pixelshuffle_block(
            feature_channels, out_channels, upscale_factor=upscale
        )

    def forward(self, x):
        out_feature = self.conv_1(x)
        # out_feature = self.stem(x)

        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)

        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)

        out_b5 = self.block_5(out_b4)
        out_b6 = self.block_6(out_b5)

        out_low_resolution = self.conv_2(out_b6) + out_feature
        return self.upsampler(out_low_resolution)
