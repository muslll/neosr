import torch
from torch import nn
from torch.nn import functional as F

from neosr.utils.registry import ARCH_REGISTRY

from .arch_util import default_init_weights, make_layer, net_opt

upscale, training = net_opt()


class Shift8(nn.Module):
    def __init__(self, groups=4, stride=1, mode="constant") -> None:
        super().__init__()
        self.g = groups
        self.mode = mode
        self.stride = stride

    def forward(self, x):
        b, c, h, w = x.shape
        out = torch.zeros_like(x)

        pad_x = F.pad(x, pad=[self.stride for _ in range(4)], mode=self.mode)
        assert c == self.g * 8

        cx, cy = self.stride, self.stride
        stride = self.stride
        out[:, 0 * self.g : 1 * self.g, :, :] = pad_x[
            :, 0 * self.g : 1 * self.g, cx - stride : cx - stride + h, cy : cy + w
        ]
        out[:, 1 * self.g : 2 * self.g, :, :] = pad_x[
            :, 1 * self.g : 2 * self.g, cx + stride : cx + stride + h, cy : cy + w
        ]
        out[:, 2 * self.g : 3 * self.g, :, :] = pad_x[
            :, 2 * self.g : 3 * self.g, cx : cx + h, cy - stride : cy - stride + w
        ]
        out[:, 3 * self.g : 4 * self.g, :, :] = pad_x[
            :, 3 * self.g : 4 * self.g, cx : cx + h, cy + stride : cy + stride + w
        ]

        out[:, 4 * self.g : 5 * self.g, :, :] = pad_x[
            :,
            4 * self.g : 5 * self.g,
            cx + stride : cx + stride + h,
            cy + stride : cy + stride + w,
        ]
        out[:, 5 * self.g : 6 * self.g, :, :] = pad_x[
            :,
            5 * self.g : 6 * self.g,
            cx + stride : cx + stride + h,
            cy - stride : cy - stride + w,
        ]
        out[:, 6 * self.g : 7 * self.g, :, :] = pad_x[
            :,
            6 * self.g : 7 * self.g,
            cx - stride : cx - stride + h,
            cy + stride : cy + stride + w,
        ]
        out[:, 7 * self.g : 8 * self.g, :, :] = pad_x[
            :,
            7 * self.g : 8 * self.g,
            cx - stride : cx - stride + h,
            cy - stride : cy - stride + w,
        ]

        # out[:, 8*self.g:, :, :] = pad_x[:, 8*self.g:, cx:cx+h, cy:cy+w]
        return out


class ResidualBlockShift(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-Shift-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockShift, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, kernel_size=1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.shift = Shift8(groups=num_feat // 8, stride=1)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.shift(self.conv1(x))))
        return identity + out * self.res_scale


class UpShiftPixelShuffle(nn.Module):
    def __init__(self, dim, scale=2) -> None:
        super().__init__()

        self.up_layer = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.LeakyReLU(0.02),
            Shift8(groups=dim // 8),
            nn.Conv2d(dim, dim * scale * scale, kernel_size=1),
            nn.PixelShuffle(upscale_factor=scale),
        )

    def forward(self, x):
        out = self.up_layer(x)
        return out


class UpShiftMLP(nn.Module):
    def __init__(self, dim, mode="bilinear", scale=2) -> None:
        super().__init__()

        self.up_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode=mode, align_corners=False),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.LeakyReLU(0.02),
            Shift8(groups=dim // 8),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        out = self.up_layer(x)
        return out


@ARCH_REGISTRY.register()
class scnet(nn.Module):
    """SCNet (https://arxiv.org/abs/2307.16140) based on the Modified SRResNet.
    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=16,
        upscale=upscale,
        **kwargs,
    ):
        super(scnet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 1)
        self.body = make_layer(ResidualBlockShift, num_block, num_feat=num_feat)

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = UpShiftMLP(num_feat, scale=self.upscale)

        elif self.upscale == 4:
            self.upconv1 = UpShiftMLP(num_feat)
            self.upconv2 = UpShiftMLP(num_feat)
        elif self.upscale == 8:
            self.upconv1 = UpShiftMLP(num_feat)
            self.upconv2 = UpShiftMLP(num_feat)
            self.upconv3 = UpShiftMLP(num_feat)
        # freeze infrence
        self.pixel_shuffle = nn.Identity()

        self.conv_hr = nn.Conv2d(num_feat, num_feat, kernel_size=1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, kernel_size=1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights(
            [self.conv_first, self.upconv1, self.conv_hr, self.conv_last], 0.1
        )
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        elif self.upscale == 8:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv3(out)))

        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        base = F.interpolate(
            x, scale_factor=self.upscale, mode="bilinear", align_corners=False
        )
        out += base
        return out


@ARCH_REGISTRY.register()
def scnet_b(**kwargs):
    return scnet(num_feat=128, num_block=64, **kwargs)


@ARCH_REGISTRY.register()
def scnet_l(**kwargs):
    return scnet(num_feat=192, num_block=96, **kwargs)
