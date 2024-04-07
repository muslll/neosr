import math
import torch
from torch import nn
from torch.nn import functional as F

from neosr.utils.registry import ARCH_REGISTRY
from .arch_util import net_opt

upscale, training = net_opt()


class Attention(nn.Module):
    def __init__(self, in_planes, ratio, K, temprature=30, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.temprature = temprature
        assert in_planes > ratio
        hidden_planes = in_planes // ratio
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes, K, kernel_size=1, bias=False),
        )

        if init_weight:
            self._initialize_weights()

    def update_temprature(self):
        if self.temprature > 1:
            self.temprature -= 1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        att = self.avgpool(x)
        att = self.net(att).view(x.shape[0], -1)
        return F.softmax(att / self.temprature, -1)


class DynamicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        grounps=1,
        bias=True,
        K=4,
        temprature=40,
        ratio=4,
        init_weight=True,
    ):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = grounps
        self.bias = bias
        self.K = K
        self.init_weight = init_weight
        self.attention = Attention(
            in_planes=in_planes,
            ratio=ratio,
            K=K,
            temprature=temprature,
            init_weight=init_weight,
        )

        self.weight = nn.Parameter(
            torch.randn(K, out_planes, in_planes // grounps, kernel_size, kernel_size),
            requires_grad=True,
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(K, out_planes), requires_grad=True)
        else:
            self.bias = None

        if self.init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.update_temprature()

    def forward(self, x):
        bs, in_planels, h, w = x.shape
        softmax_att = self.attention(x)
        #x = x.view(1, -1, h, w)
        x = x.reshape(1, -1, h, w)
        weight = self.weight.view(self.K, -1)
        aggregate_weight = torch.mm(softmax_att, weight).view(
            bs * self.out_planes,
            self.in_planes // self.groups,
            self.kernel_size,
            self.kernel_size,
        )

        if self.bias is not None:
            bias = self.bias.view(self.K, -1)  # K,out_p
            aggregate_bias = torch.mm(softmax_att, bias).view(-1)  # bs,out_p
            output = F.conv2d(
                x,
                weight=aggregate_weight,
                bias=aggregate_bias,
                stride=self.stride,
                padding=self.padding,
                groups=self.groups * bs,
                dilation=self.dilation,
            )
        else:
            output = F.conv2d(
                x,
                weight=aggregate_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                groups=self.groups * bs,
                dilation=self.dilation,
            )

        output = output.view(bs, self.out_planes, h, w)
        return output


class Block(nn.Module):  # EBs
    def __init__(self):
        super(Block, self).__init__()
        features = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=features,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            DynamicConv(
                in_planes=features, out_planes=features, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv6 = nn.Sequential(
            DynamicConv(
                in_planes=features, out_planes=features, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv9 = nn.Sequential(
            DynamicConv(
                in_planes=features, out_planes=features, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv12 = nn.Sequential(
            DynamicConv(
                in_planes=features, out_planes=features, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv15 = nn.Sequential(
            DynamicConv(
                in_planes=features, out_planes=features, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x4_1 = x4 + x1
        x5 = self.conv5(x4_1)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x7_1 = x7 + x4_1
        x8 = self.conv8(x7_1)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x10_1 = x10 + x7_1
        x11 = self.conv11(x10_1)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x13_1 = x13 + x10_1
        x14 = self.conv14(x13_1)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x16_1 = x16 + x13_1
        return x16_1


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class MeanShift(nn.Conv2d):
    def __init__(
        self,
        rgb_range,
        rgb_mean=(0.5, 0.5, 0.5),
        rgb_std=(1.0, 1.0, 1.0),
        sign=-1,
    ):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.tensor(rgb_mean) / std
        if not training:
            for p in self.parameters():
                p.requires_grad = False


def init_weights(modules):
    pass


class _UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [
                    nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group)
                ]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)
        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, multi_scale, group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


@ARCH_REGISTRY.register()
class hdsrnet(nn.Module):
    def __init__(
        self, features=64, multi_scale=False, upscale=upscale, conv=default_conv
    ):
        super(hdsrnet, self).__init__()
        self.scale = upscale
        self.conv1 = nn.Sequential(  # SN
            nn.Conv2d(
                in_channels=3,
                out_channels=features,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv17 = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=3,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False,
            )
        )

        self.ReLU = nn.ReLU(inplace=True)
        self.upsample = UpsampleBlock(
            64, scale=self.scale, multi_scale=multi_scale, group=1
        )

        self.Block = Block()
        self.sub_mean = MeanShift(1.0)
        self.add_mean = MeanShift(1.0, sign=1)

    def forward(self, x):
        x = self.sub_mean(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x9_1 = x8 + x9
        x10 = self.conv10(x9_1)
        x10_1 = x10 + x7
        x11 = self.conv11(x10_1)
        x11_1 = x11 + x6
        x12 = self.conv12(x11_1)
        x12_1 = x12 + x5
        x13 = self.conv13(x12_1)
        x13_1 = x13 + x4
        x14 = self.conv14(x13_1)
        x14_1 = x14 + x3
        x15 = self.conv15(x14_1)
        x15_1 = x15 + x2
        x16 = self.conv16(x15_1)
        x16_1 = x16 + x1

        x16_2 = self.Block(x)

        x16_3 = x16_2 * x16_1
        temp = self.upsample(x16_3, scale=self.scale)
        # temp=self.tail(x16_3)
        x17 = self.conv17(temp)
        # out = self.add_mean(temp)
        out = self.add_mean(x17)

        return out

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, DynamicConv):
                m.update_temperature()
