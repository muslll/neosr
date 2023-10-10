import math
import torch
from pathlib import Path

from torch import nn as nn
from neosr.utils.registry import ARCH_REGISTRY
from neosr.utils.options import parse_options

# initialize options parsing
root_path = Path(__file__).parents[2]
opt, args = parse_options(root_path, is_train=True)
# set scale factor in network parameters
upscale = opt['scale']


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

@ARCH_REGISTRY.register()
class mdbn(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, upscale=upscale, res_scale=1.0):
        super(mdbn, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlock, num_block, num_feat=num_feat, res_scale=res_scale)

        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.upsample = Upsample(upscale, num_feat)

        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x
        x = self.conv_last(self.upsample(res))

        return x

class ResidualBlock(nn.Module):
    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlock, self).__init__()
        self.res_scale = res_scale
        self.baseblock1 = BaseBlock(num_feat)
        self.baseblock2 = BaseBlock(num_feat)

    def forward(self, x):
        identity = x

        x = self.baseblock1(x)
        x = self.baseblock2(x)

        return identity + x * self.res_scale

class BaseBlock(nn.Module):
    def __init__(self, num_feat):
        super(BaseBlock, self).__init__()
        self.uconv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.uconv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.dconv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.uconv2(self.act(self.uconv1(x)))
        x2 = self.dconv(x)
        x = self.act(x1 + x2)
        return x

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

