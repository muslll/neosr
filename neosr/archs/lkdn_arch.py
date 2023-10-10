import torch
from torch import nn as nn
from torch.nn import functional as F
from pathlib import Path

from neosr.archs.arch_util import default_init_weights
from neosr.utils.registry import ARCH_REGISTRY
from neosr.utils.options import parse_options

# initialize options parsing
root_path = Path(__file__).parents[2]
opt, args = parse_options(root_path, is_train=True)
# set scale factor in network parameters
upscale = opt['scale']

class BSConvU(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=True,
                 padding_mode="zeros"):
        super().__init__()

        # pointwise
        self.pw = nn.Conv2d(
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
        self.dw = nn.Conv2d(
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


class BSConvU_idt(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=True,
                 padding_mode="zeros"):
        super().__init__()

        # pointwise
        self.pw = nn.Conv2d(
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
        self.dw = nn.Conv2d(
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

    def forward(self, x):
        fea = self.pw(x)
        fea = self.dw(fea)
        return fea + x


class BSConvU_rep(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=True,
                 padding_mode="zeros"):
        super().__init__()

        # pointwise
        self.pw = nn.Conv2d(
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
        self.dw = nn.Conv2d(
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

        self.rep1x1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea) + fea
        fea = self.dw(fea) + fea + self.rep1x1(fea)
        return fea


class Attention(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.pointwise = nn.Conv2d(dim, dim, 1)
        self.depthwise = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.depthwise_dilated = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)

    def forward(self, x):
        u = x.clone()
        attn = self.pointwise(x)
        attn = self.depthwise(attn)
        attn = self.depthwise_dilated(attn)
        return u * attn


class LKDB(nn.Module):

    def __init__(self, in_channels, out_channels, atten_channels=None, conv=nn.Conv2d):
        super().__init__()

        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        if (atten_channels is None):
            self.atten_channels = in_channels
        else:
            self.atten_channels = atten_channels

        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = conv(in_channels, self.rc, kernel_size=3, padding=1)
        self.c2_d = nn.Conv2d(self.rc, self.dc, 1)
        self.c2_r = conv(self.rc, self.rc, kernel_size=3, padding=1)
        self.c3_d = nn.Conv2d(self.rc, self.dc, 1)
        self.c3_r = conv(self.rc, self.rc, kernel_size=3, padding=1)

        self.c4 = BSConvU(self.rc, self.dc, kernel_size=3, padding=1)
        self.act = nn.GELU()

        self.c5 = nn.Conv2d(self.dc * 4, self.atten_channels, 1)
        self.atten = Attention(self.atten_channels)
        self.c6 = nn.Conv2d(self.atten_channels, out_channels, 1)
        self.pixel_norm = nn.LayerNorm(out_channels)  # channel-wise
        default_init_weights([self.pixel_norm], 0.1)

    def forward(self, input):

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)

        out_fused = self.atten(out)
        out_fused = self.c6(out_fused)
        out_fused = out_fused.permute(0, 2, 3, 1)  # (B, H, W, C)
        out_fused = self.pixel_norm(out_fused)
        out_fused = out_fused.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        return out_fused + input


def UpsampleOneStep(in_channels, out_channels, upscale_factor=4):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])


class Upsampler_rep(nn.Module):

    def __init__(self, in_channels, out_channels, upscale_factor=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 3, 1, 1)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels * 2, 1)
        self.conv3x3 = nn.Conv2d(in_channels * 2, out_channels * (upscale_factor**2), 3)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        v1 = F.conv2d(x, self.conv1x1.weight, self.conv1x1.bias, padding=0)
        v1 = F.pad(v1, (1, 1, 1, 1), 'constant', 0)
        b0_pad = self.conv1x1.bias.view(1, -1, 1, 1)
        v1[:, :, 0:1, :] = b0_pad
        v1[:, :, -1:, :] = b0_pad
        v1[:, :, :, 0:1] = b0_pad
        v1[:, :, :, -1:] = b0_pad
        v2 = F.conv2d(v1, self.conv3x3.weight, self.conv3x3.bias, padding=0)
        out = self.conv1(x) + self.conv3(x) + v2
        return self.pixel_shuffle(out)

@ARCH_REGISTRY.register()
class lkdn(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=56,
                 num_atten=56,
                 num_block=8,
                 upscale=upscale,
                 num_in=4,
                 conv='BSConvU',
                 upsampler='pixelshuffledirect'):
        super().__init__()
        self.num_in = num_in
        if conv == 'BSConvU_idt':
            self.conv = BSConvU_idt
        elif conv == 'BSConvU_rep':
            self.conv = BSConvU_rep
        elif conv == 'BSConvU':
            self.conv = BSConvU
        else:
            raise NotImplementedError(f'conv {conv} is not supported yet.')
        print(conv)
        self.fea_conv = BSConvU(num_in_ch * num_in, num_feat, kernel_size=3, padding=1)

        self.B1 = LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)
        self.B2 = LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)
        self.B3 = LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)
        self.B4 = LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)
        self.B5 = LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)
        self.B6 = LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)
        self.B7 = LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)
        self.B8 = LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        self.GELU = nn.GELU()

        self.c2 = BSConvU(num_feat, num_feat, kernel_size=3, padding=1)

        if upsampler == 'pixelshuffledirect':
            self.upsampler = UpsampleOneStep(num_feat, num_out_ch, upscale_factor=upscale)
        elif upsampler == 'pixelshuffle_rep':
            self.upsampler = Upsampler_rep(num_feat, num_out_ch, upscale_factor=upscale)
        else:
            raise NotImplementedError("Check the Upsampler. None or not support yet.")

    def forward(self, input):
        input = torch.cat([input] * self.num_in, dim=1)
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)

        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8], dim=1)
        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)

        out_lr = self.c2(out_B) + out_fea
        output = self.upsampler(out_lr)

        return output

@ARCH_REGISTRY.register()
class lkdn_s(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=42,
                 num_atten=42,
                 num_block=5,
                 upscale=upscale,
                 num_in=4,
                 conv='BSConvU',
                 upsampler='pixelshuffledirect'):
        super().__init__()
        self.num_in = num_in
        if conv == 'BSConvU_idt':
            self.conv = BSConvU_idt
        elif conv == 'BSConvU_rep':
            self.conv = BSConvU_rep
        elif conv == 'BSConvU':
            self.conv = BSConvU
        else:
            raise NotImplementedError(f'conv {conv} is not supported yet.')
        print(conv)
        self.fea_conv = BSConvU(num_in_ch * num_in, num_feat, kernel_size=3, padding=1)

        self.B1 = LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)
        self.B2 = LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)
        self.B3 = LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)
        self.B4 = LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)
        self.B5 = LKDB(in_channels=num_feat, out_channels=num_feat, atten_channels=num_atten, conv=self.conv)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        self.GELU = nn.GELU()

        self.c2 = BSConvU(num_feat, num_feat, kernel_size=3, padding=1)

        if upsampler == 'pixelshuffledirect':
            self.upsampler = UpsampleOneStep(num_feat, num_out_ch, upscale_factor=upscale)
        elif upsampler == 'pixelshuffle_rep':
            self.upsampler = Upsampler_rep(num_feat, num_out_ch, upscale_factor=upscale)
        else:
            raise NotImplementedError("Check the Upsampler. None or not support yet.")

    def forward(self, input):
        input = torch.cat([input] * self.num_in, dim=1)
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)

        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5], dim=1)
        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)

        out_lr = self.c2(out_B) + out_fea
        output = self.upsampler(out_lr)

        return output
