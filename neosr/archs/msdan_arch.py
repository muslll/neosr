# type: ignore  # noqa: PGH003
import torch
import torch.nn
from torch import nn as nn
from torch.nn import functional as F

from neosr.archs.arch_util import net_opt
from neosr.utils.registry import ARCH_REGISTRY

upscale, __ = net_opt()


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
    ----
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
    -------
        nn.Sequential: Stacked blocks in nn.Sequential.

    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class BSConvU(torch.nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=1,
        stride=1,
        dilation=1,
        bias=True,
        padding_mode="zeros",
        with_bn=False,
        bn_kwargs=None,
    ):
        super().__init__()
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}
        # pointwise
        self.add_module(
            "pw",
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
            ),
        )
        # batchnorm
        if with_bn:
            self.add_module(
                "bn", torch.nn.BatchNorm2d(num_features=out_channels, **bn_kwargs)
            )
        # depthwise
        self.add_module(
            "dw",
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
            ),
        )


class EMSSA(nn.Module):
    def __init__(self, channels):
        super(EMSSA, self).__init__()
        self.BSConv3 = BSConvU(channels // 4, channels // 4, kernel_size=3, padding=1)
        self.BSConv5 = BSConvU(channels // 4, channels // 4, kernel_size=5, padding=2)
        self.BSConv7 = BSConvU(channels // 4, channels // 4, kernel_size=7, padding=3)
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv1_mid = nn.Conv2d(
            in_channels=channels // 4,
            out_channels=channels // 4,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv1_down = nn.Conv2d(
            in_channels=channels,
            out_channels=channels // 4,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv1_up = nn.Conv2d(
            in_channels=channels // 4,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x0):
        h = x0.size(2)
        w = x0.size(3)
        x = self.conv1_down(x0)
        down_size2 = (h // 2, w // 2)
        down_size4 = (h // 4, w // 4)
        down_size8 = (h // 8, w // 8)
        s0 = self.conv1_mid(x)
        s1 = F.adaptive_max_pool2d(x, down_size2)
        s1 = self.BSConv7(s1)
        s1 = F.interpolate(s1, size=(h, w), mode="nearest")
        s2 = F.adaptive_max_pool2d(x, down_size4)
        s2 = self.BSConv5(s2)
        s2 = F.interpolate(s2, size=(h, w), mode="nearest")
        s3 = F.adaptive_max_pool2d(x, down_size8)
        s3 = self.BSConv3(s3)
        s3 = F.interpolate(s3, size=(h, w), mode="nearest")
        out = torch.add(
            self.conv1(torch.cat((s0, s1, s2, s3), dim=1)), self.conv1_up(x)
        )
        out = self.sigmoid(self.conv1(out))
        out = x0 * out
        return out


class SAFM(nn.Module):
    def __init__(self, channels, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = channels // n_levels
        # Spatial Weighting
        self.mfr = nn.ModuleList([
            nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim)
            for i in range(self.n_levels)
        ])
        # # Feature Aggregation
        self.aggr = nn.Conv2d(channels, channels, 1, 1, 0)
        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]
        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2**i, w // 2**i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode="nearest")
            else:
                s = self.mfr[i](xc[i])
            out.append(s)
        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out


class ESA(nn.Module):
    def __init__(self, channels):
        super(ESA, self).__init__()
        f = channels // 4
        self.conv1 = nn.Conv2d(channels, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(
            c3, (x.size(2), x.size(3)), mode="bilinear", align_corners=False
        )
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class EBFB(nn.Module):
    def __init__(self, channels):
        super(EBFB, self).__init__()
        self.BSConv3 = BSConvU(channels, channels, kernel_size=3, padding=1)
        self.BSConv5 = BSConvU(channels, channels, kernel_size=5, padding=2)
        self.BSConv7 = BSConvU(channels, channels, kernel_size=7, padding=3)
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv1_down = nn.Conv2d(
            in_channels=channels * 4,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.GELU = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.mul(self.sigmoid(self.conv1(x)), self.GELU(self.BSConv3(x)))
        out1 = torch.add(x1, x2)
        x3 = self.conv1(out1)
        x4 = torch.mul(self.sigmoid(self.conv1(out1)), self.GELU(self.BSConv5(out1)))
        out2 = torch.add(x3, x4)
        x5 = self.conv1(out2)
        x6 = torch.mul(self.sigmoid(self.conv1(out2)), self.GELU(self.BSConv7(out2)))
        out = self.conv1_down(torch.cat((x1, x3, x5, x6), 1))
        return out


class EBFB_SRB(nn.Module):
    def __init__(self, channels):
        super(EBFB_SRB, self).__init__()
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv1_down = nn.Conv2d(
            in_channels=channels * 4,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.ReLU(torch.add(x, self.conv3(x)))
        out1 = torch.add(x1, x2)
        x3 = self.conv1(out1)
        x4 = self.ReLU(torch.add(out1, self.conv3(out1)))
        out2 = torch.add(x3, x4)
        x5 = self.conv1(out2)
        x6 = self.ReLU(torch.add(out2, self.conv3(out2)))
        out = self.conv1_down(torch.cat((x1, x3, x5, x6), 1))
        return out


class RFDB_WOA(nn.Module):
    def __init__(self, channels):
        super(RFDB_WOA, self).__init__()
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3_half = nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv1_half = nn.Conv2d(
            in_channels=channels,
            out_channels=channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv1_down = nn.Conv2d(
            in_channels=channels * 2,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):  # 48 channels
        x1 = self.ReLU(self.conv1_half(x))  # 24 channels
        x2 = self.ReLU(torch.add(x, self.conv3(x)))  # 48 channels
        x3 = self.ReLU(self.conv1_half(x2))  # 24 channels
        x4 = self.ReLU(torch.add(x2, self.conv3(x2)))  # 48 channels
        x5 = self.ReLU(self.conv1_half(x4))  # 24 channels
        x6 = self.ReLU(
            self.conv3_half(self.ReLU(torch.add(x4, self.conv3(x4))))
        )  # 24 channels
        out = torch.cat((x1, x3, x5, x6), 1)  # 96 channels
        out = self.ReLU(self.conv1_down(out))  # 48 channels
        return out


class RFDB_PAB(nn.Module):
    def __init__(self, channels):
        super(RFDB_PAB, self).__init__()
        self.BSConv3 = BSConvU(channels, channels, kernel_size=3, padding=1)
        self.conv3_half = nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv1_half = nn.Conv2d(
            in_channels=channels,
            out_channels=channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv1_down = nn.Conv2d(
            in_channels=channels * 2,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.ReLU = nn.ReLU(inplace=True)
        self.GELU = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # 48 channels
        x1 = self.ReLU(self.conv1_half(x))  # 24 channels
        x2 = torch.mul(
            self.sigmoid(self.conv1(x)), self.GELU(self.BSConv3(x))
        )  # 48 channels
        x3 = self.ReLU(self.conv1_half(x2))  # 24 channels
        x4 = torch.mul(
            self.sigmoid(self.conv1(x2)), self.GELU(self.BSConv3(x2))
        )  # 48 channels
        x5 = self.ReLU(self.conv1_half(x4))  # 24 channels
        x6 = self.ReLU(
            self.conv3_half(
                torch.mul(self.sigmoid(self.conv1(x4)), self.GELU(self.BSConv3(x4)))
            )
        )  # 24 channels
        out = torch.cat((x1, x3, x5, x6), 1)  # 96 channels
        out = self.ReLU(self.conv1_down(out))  # 48 channels
        return out


class upsampler(nn.Module):
    def __init__(self, channels, upscale_factor, mid_channels=54):
        super(upsampler, self).__init__()
        self.BSConv_2 = BSConvU(channels, channels, kernel_size=3, padding=1)  # x2
        self.BSConv1_2 = BSConvU(
            channels // 4, channels, kernel_size=3, padding=1
        )  # x2
        self.BSConv_3 = BSConvU(channels, mid_channels, kernel_size=3, padding=1)  # x3
        self.BSConv1_3 = BSConvU(
            mid_channels // 9, channels, kernel_size=3, padding=1
        )  # x3
        self.up_2 = nn.PixelShuffle(upscale_factor=2)
        self.up_3 = nn.PixelShuffle(upscale_factor=3)
        self.GELU = nn.GELU()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        if self.upscale_factor == 2:
            x = self.BSConv_2(x)
            x = self.up_2(x)
            x = self.BSConv1_2(x)
            x = self.GELU(x)
            return x
        if self.upscale_factor == 3:
            x = self.BSConv_3(x)
            x = self.up_3(x)
            x = self.BSConv1_3(x)
            x = self.GELU(x)
            return x
        if self.upscale_factor == 4:
            x = self.BSConv_2(x)
            x = self.up_2(x)
            x = self.BSConv1_2(x)
            x = self.GELU(x)
            x = self.BSConv_2(x)
            x = self.up_2(x)
            x = self.BSConv1_2(x)
            x = self.GELU(x)
            return x


class DFEB(nn.Module):
    def __init__(self, channels):
        super(DFEB, self).__init__()

        self.EBFB = EBFB(channels)
        # self.EBFB_SRB = EBFB_SRB(channels)
        # self.RFDB_WOA = RFDB_WOA(channels)
        # self.RFDB_PAB = RFDB_PAB(channels)

        # self.ESA = ESA(channels)
        self.EMSSA = EMSSA(channels)
        # self.SAFM = SAFM(channels)

    def forward(self, x0):
        x = self.EBFB(x0)
        # x = self.EBFB_SRB(x0)
        # x = self.RFDB_WOA(x0)
        # x = self.RFDB_PAB(x0)

        x = self.EMSSA(x)
        # x = self.SAFM(x)
        # x = self.ESA(x)
        x = torch.add(x, x0)
        return x


@ARCH_REGISTRY.register()
class msdan(nn.Module):
    def __init__(self, channels=48, num_DFEB=8, upscale_factor=upscale):
        super(msdan, self).__init__()
        self.layers = make_layer(
            basic_block=DFEB, num_basic_block=num_DFEB, channels=channels
        )
        self.BSConv_first = BSConvU(3, channels, kernel_size=3, padding=1)
        self.upsampler = upsampler(channels, upscale_factor)
        self.BSConv_last = BSConvU(channels, 3, kernel_size=3, padding=1)
        self.upscale_factor = upscale_factor

    def forward(self, x0):
        x1 = self.BSConv_first(x0)
        x = self.layers(x1)
        x = torch.add(x, x1)
        if self.upscale_factor == 2:
            x = self.upsampler(x)  # x2
            x_up = F.interpolate(x0, scale_factor=2, mode="bicubic")  # x2
        if self.upscale_factor == 3:
            x = self.upsampler(x)  # x3
            x_up = F.interpolate(x0, scale_factor=3, mode="bicubic")  # x3
        if self.upscale_factor == 4:
            x = self.upsampler(x)  # x4
            x_up = F.interpolate(x0, scale_factor=4, mode="bicubic")  # x4
        out = torch.add(x_up, self.BSConv_last(x))
        return out
