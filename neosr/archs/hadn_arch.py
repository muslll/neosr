import torch
import torch.nn.functional as F
from torch import nn

from neosr.utils.registry import ARCH_REGISTRY
from .arch_util import net_opt

upscale, training = net_opt()


class BSConvU(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
        padding_mode="zeros",
        with_ln=False,
        bn_kwargs=None,
    ):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


def stdv_channels(F):
    assert F.dim() == 4
    F_mean = mean_channels(F)
    eps = 1e-7
    F_variance = (F - F_mean + eps).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (
        F.size(2) * F.size(3)
    )
    return F_variance.pow(0.5)


def mean_channels(F):
    assert F.dim() == 4
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CrossCCA(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(CrossCCA, self).__init__()

        self.conv3_1_A = nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0))
        self.conv3_1_B = nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1))
        self.cca = CCALayer(out_channels)
        self.depthwise = nn.Conv2d(
            out_channels, out_channels, 5, padding=2, groups=out_channels,
        )
        self.depthwise_dilated = nn.Conv2d(
            out_channels,
            out_channels,
            5,
            stride=1,
            padding=6,
            groups=out_channels,
            dilation=3,
        )
        self.conv = nn.Conv2d(out_channels, out_channels, 1, padding=0)
        self.active = nn.Sigmoid()

    def forward(self, input):
        x = self.conv3_1_A(input) + self.conv3_1_B(input)
        x_cca = self.cca(x)
        x_de = self.depthwise(x_cca + input)
        x_de = self.depthwise_dilated(x_de)
        x_fea = x_de + x
        x_fea = self.active(self.conv(x_fea))

        return x_fea * input


class ESDB(nn.Module):
    def __init__(self, in_channels, out_channels, conv=nn.Conv2d, p=0.25):
        super(ESDB, self).__init__()
        kwargs = {"padding": 1}

        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c1_d = nn.Conv2d(in_channels * 2, self.rc, 1, groups=2)
        self.c1_r = conv(in_channels, self.rc, kernel_size=3, **kwargs)

        self.c2_d = nn.Conv2d(self.remaining_channels * 2, self.rc, 1, groups=2)
        self.c2_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)

        self.c3_d = nn.Conv2d(self.remaining_channels * 2, self.rc, 1, groups=2)
        self.c3_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)

        self.c4 = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)
        self.act = nn.GELU()

        self.CrossCCA = CrossCCA(in_channels, out_channels)

    def forward(self, input):
        r_c1 = self.c1_r(input)
        r_c1 = self.act(r_c1 + input)
        r_c1_cat = self.c1_d(torch.cat([input, r_c1], dim=1))

        r_c2 = self.c2_r(r_c1_cat)
        r_c2 = self.act(r_c2 + r_c1_cat)
        r_c2_cat = self.c2_d(torch.cat([r_c1_cat, r_c2], dim=1))

        r_c3 = self.c3_r(r_c2_cat)
        r_c3 = self.act(r_c3 + r_c2_cat)
        r_c3_cat = self.c3_d(torch.cat([r_c2_cat, r_c3], dim=1))
        r_c4 = self.c4(r_c3_cat)

        out_fused = self.CrossCCA(r_c4)

        return out_fused + input


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
    ----
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)


class PixelShuffleDirect(nn.Module):
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        super(PixelShuffleDirect, self).__init__()
        self.upsampleOneStep = UpsampleOneStep(
            scale, num_feat, num_out_ch, input_resolution=None,
        )

    def forward(self, x):
        return self.upsampleOneStep(x)


class PixelShuffleBlcok(nn.Module):
    def __init__(self, in_feat, num_feat, num_out_ch):
        super(PixelShuffleBlcok, self).__init__()
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(in_feat, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
            nn.PixelShuffle(2),
        )
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))
        return x


class NearestConv(nn.Module):
    def __init__(self, in_ch, num_feat, num_out_ch, conv=nn.Conv2d):
        super(NearestConv, self).__init__()
        self.conv_before_upsample = nn.Sequential(
            conv(in_ch, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True),
        )
        self.conv_up1 = conv(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = conv(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = conv(num_feat, num_feat, 3, 1, 1)
        self.conv_last = conv(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv_before_upsample(x)
        x = self.lrelu(
            self.conv_up1(
                torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest"),
            ),
        )
        x = self.lrelu(
            self.conv_up2(
                torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest"),
            ),
        )
        x = self.conv_last(self.lrelu(self.conv_hr(x)))
        return x


class PA(nn.Module):
    """PA is pixel attention"""

    def __init__(self, nf):
        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


class PA_UP(nn.Module):
    def __init__(self, nf, unf, out_nc, scale=4, conv=nn.Conv2d):
        super(PA_UP, self).__init__()
        self.upconv1 = conv(nf, unf, 3, 1, 1, bias=True)
        self.att1 = PA(unf)
        self.HRconv1 = conv(unf, unf, 3, 1, 1, bias=True)

        if scale == 4:
            self.upconv2 = conv(unf, unf, 3, 1, 1, bias=True)
            self.att2 = PA(unf)
            self.HRconv2 = conv(unf, unf, 3, 1, 1, bias=True)

        self.conv_last = conv(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, fea):
        fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode="nearest"))
        fea = self.lrelu(self.att1(fea))
        fea = self.lrelu(self.HRconv1(fea))
        fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode="nearest"))
        fea = self.lrelu(self.att2(fea))
        fea = self.lrelu(self.HRconv2(fea))
        fea = self.conv_last(fea)
        return fea


@ARCH_REGISTRY.register()
class hadn(nn.Module):
    def __init__(
        self,
        num_in_ch=3,
        num_feat=64,
        num_block=8,
        num_out_ch=3,
        upscale=upscale,
        conv="BSConvU",
        upsampler="pixelshuffledirect",
        p=0.25,
    ):
        super(hadn, self).__init__()
        kwargs = {"padding": 1}
        if conv == "BSConvU":
            self.conv = BSConvU
        else:
            self.conv = nn.Conv2d
        self.fea_conv = self.conv(num_in_ch * 4, num_feat, kernel_size=3, **kwargs)

        self.B1 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B2 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B3 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B4 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B5 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B6 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B7 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B8 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        self.GELU = nn.GELU()
        self.up = upscale

        self.c2 = self.conv(num_feat, num_feat, kernel_size=3, **kwargs)
        if upsampler == "pixelshuffledirect":
            self.upsampler = PixelShuffleDirect(
                scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch,
            )
        elif upsampler == "pixelshuffleblock":
            self.upsampler = PixelShuffleBlcok(
                in_feat=num_feat, num_feat=num_feat, num_out_ch=num_out_ch,
            )
        elif upsampler == "nearestconv":
            self.upsampler = NearestConv(
                in_ch=num_feat, num_feat=num_feat, num_out_ch=num_out_ch,
            )
        elif upsampler == "pa":
            self.upsampler = PA_UP(nf=num_feat, unf=24, out_nc=num_out_ch)
        else:
            raise NotImplementedError("Check the Upsampeler. None or not support yet")

    def forward(self, input):
        input_cat = torch.cat([input, input, input, input], dim=1)
        out_fea = self.fea_conv(input_cat)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)

        trunk = torch.cat(
            [out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8], dim=1,
        )

        out_B = self.c1(trunk)

        out_B = self.GELU(out_B)

        out_lr = self.c2(out_B) + out_fea

        input_up = F.interpolate(
            input,
            (input.size(2) * self.up, input.size(3) * self.up),
            mode="bilinear",
            align_corners=False,
        )

        output = self.upsampler(out_lr)

        return output + input_up
