from torch import nn as nn
import torch

from neosr.utils.registry import ARCH_REGISTRY
from .arch_util import net_opt

upscale, training = net_opt()


@ARCH_REGISTRY.register()
class patch(nn.Module):
    """VGG style discriminator with input size 128 x 128 or 256 x 256.

    From https://github.com/Suanmd/Patch-Loss-for-Super-Resolution/

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.Default: 64.
    """

    def __init__(self, num_in_ch=3, num_feat=64, input_size=128):
        super(patch, self).__init__()
        self.input_size = input_size
        if upscale == 4:
            assert self.input_size == 128 or self.input_size == 256, (
                f'input size must be 128 or 256, but received {input_size}')
        elif upscale == 2:
            assert self.input_size == 64 or self.input_size == 128 or self.input_size == 256, (
                f'input size must be 64, 128, or 256, but received {input_size}')
        elif upscale == 1:
            assert self.input_size == 32 or self.input_size == 64 or self.input_size == 128 or self.input_size == 256, (
                f'input size must be 32, 64, 128, or 256, but received {input_size}')
        else:
            raise ValueError('Patch discriminator only supports 1x, 2x, and 4x models.')

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        if upscale == 2 or self.input_size == 64:
            self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
            self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
            self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
            self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

            if upscale == 4 or self.input_size == 128:
                self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
                self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
                self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
                self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

                if self.input_size == 256:
                    self.conv5_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
                    self.bn5_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
                    self.conv5_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
                    self.bn5_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == self.input_size, (f'Input size must be identical to input_size, but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # output spatial size: /2

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # output spatial size: /4

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # output spatial size: /8

        if upscale == 2 or self.input_size == 64:
            feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
            feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: /16

            if upscale == 4 or self.input_size == 128:
                feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
                feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: /32

                if self.input_size == 256:
                    feat = self.lrelu(self.bn5_0(self.conv5_0(feat)))
                    feat = self.lrelu(self.bn5_1(self.conv5_1(feat)))  # output spatial size: / 64

        # spatial size: (4, 4)
        feat = feat.contiguous()
        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out
