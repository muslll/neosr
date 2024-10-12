# type: ignore
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from torch import nn

from neosr.archs.arch_util import net_opt
from neosr.utils.registry import ARCH_REGISTRY

upscale, __ = net_opt()


def pad_tensor(t, pattern):
    pattern = pattern.view(1, -1, 1, 1)
    t = F.pad(t, (1, 1, 1, 1), "constant", 0)
    t[:, :, 0:1, :] = pattern
    t[:, :, -1:, :] = pattern
    t[:, :, :, 0:1] = pattern
    t[:, :, :, -1:] = pattern

    return t


class MBConv(nn.Module):
    def __init__(self, n_feat, ratio=2):
        super().__init__()
        i_feat = n_feat * ratio
        self.expand_conv = nn.Conv2d(n_feat, i_feat, 1, 1, 0)
        self.fea_conv = nn.Conv2d(i_feat, i_feat, 3, 1, 0)
        self.reduce_conv = nn.Conv2d(i_feat, n_feat, 1, 1, 0)
        self.se = ASR(i_feat)

    def forward(self, x):
        out = self.expand_conv(x)
        out_identity = out

        # explicitly padding with bias for reparameterizing in the test phase
        b0 = self.expand_conv.bias
        out = pad_tensor(out, b0)
        out = self.fea_conv(out)
        out = self.se(out) + out_identity
        out = self.reduce_conv(out)
        return out + x

    def switch_to_deploy(self):
        n_feat, _, _, _ = self.reduce_conv.weight.data.shape
        self.conv = nn.Conv2d(n_feat, n_feat, 3, 1, 1)

        k0 = self.expand_conv.weight.data
        b0 = self.expand_conv.bias.data

        k1 = self.fea_conv.weight.data
        b1 = self.fea_conv.bias.data

        k2 = self.reduce_conv.weight.data
        b2 = self.reduce_conv.bias.data

        # first step: remove the ASR
        a = self.se.se(self.se.tensor)

        k1 = k1 * (a.permute(1, 0, 2, 3))
        b1 = b1 * (a.view(-1))

        # second step: remove the middle identity
        for i in range(2 * n_feat):
            k1[i, i, 1, 1] += 1.0

        # third step: merge the first 1x1 convolution and the next 3x3 convolution
        merge_k0k1 = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
        merge_b0b1 = b0.view(1, -1, 1, 1) * torch.ones(1, 2 * n_feat, 3, 3)  # .cuda()
        merge_b0b1 = F.conv2d(input=merge_b0b1, weight=k1, bias=b1)

        # third step: merge the remain 1x1 convolution
        merge_k0k1k2 = F.conv2d(
            input=merge_k0k1.permute(1, 0, 2, 3), weight=k2
        ).permute(1, 0, 2, 3)
        merge_b0b1b2 = F.conv2d(input=merge_b0b1, weight=k2, bias=b2).view(-1)

        # last step: remove the global identity
        for i in range(n_feat):
            merge_k0k1k2[i, i, 1, 1] += 1.0

        self.conv.weight.data = merge_k0k1k2.float()
        self.conv.bias.data = merge_b0b1b2.float()

        for para in self.parameters():
            para.detach_()

        self.__delattr__("expand_conv")
        self.__delattr__("fea_conv")
        self.__delattr__("reduce_conv")
        self.__delattr__("se")

        # redundancy
        delattr(self, "expand_conv")
        delattr(self, "fea_conv")
        delattr(self, "reduce_conv")
        delattr(self, "se")
        del self._modules["expand_conv"]
        del self._parameters["expand_conv"]
        del self._modules["fea_conv"]
        del self._parameters["fea_conv"]
        del self._modules["reduce_conv"]
        del self._parameters["reduce_conv"]
        del self._modules["se"]
        del self._parameters["se"]



class ASR(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.n_feat = n_feat
        self.tensor = nn.Parameter(
            0.1 * torch.ones((1, n_feat, 1, 1)), requires_grad=True
        )
        self.se = nn.Sequential(
            Reduce("b c 1 1 -> b c", "mean"),
            nn.Linear(n_feat, n_feat // 4, bias=False),
            nn.SiLU(),
            nn.Linear(n_feat // 4, n_feat, bias=False),
            nn.Sigmoid(),
            Rearrange("b c -> b c 1 1"),
        )
        self.init_weights()

    def init_weights(self):
        # to make sure the inital [0.5,0.5,...,0.5]
        self.se[1].weight.data.fill_(1)
        self.se[3].weight.data.fill_(1)

    def forward(self, x):
        attn = self.se(self.tensor)
        return attn * x


class Block(nn.Module):
    def __init__(self, n_feat, version, f=16):
        super().__init__()
        self.f = f
        self.version = version

        if self.version == 1:
            self.body = nn.Sequential(
                self._conv_or_mb(n_feat),
                nn.LeakyReLU(0.05, inplace=True),
                self._conv_or_mb(n_feat),
            )
        elif self.version == 2:
            self.body = nn.Sequential(
                self._conv_or_mb(n_feat),
                nn.LeakyReLU(0.05, inplace=True),
                self._conv_or_mb(n_feat),
                self._conv_or_pconv(n_feat),
                LocalAttention(n_feat, f),
            )
        elif self.version == 3:
            self.body = nn.Sequential(
                self._conv_or_mb(n_feat),
                nn.LeakyReLU(0.05, inplace=True),
                self._conv_or_mb(n_feat),
                self._conv_or_pconv(n_feat),
                LocalAttention(n_feat, f, speed=True),
            )
        else:
            self.body = nn.Sequential(
                self._conv_or_mb(n_feat),
                nn.LeakyReLU(0.05, inplace=True),
                self._conv_or_mb(n_feat),
                nn.LeakyReLU(0.05, inplace=True),
                self._conv_or_mb(n_feat),
                self._conv_or_pconv(n_feat),
                LocalAttention(n_feat, f),
            )

    def forward(self, x):
        return self.body(x)

    def _conv_or_mb(self, n_feat):
        if self.training:
            return MBConv(n_feat)
        return nn.Conv2d(n_feat, n_feat, 3, 1, 1)

    def _conv_or_pconv(self, n_feat):
        if self.training:
            return PConv(n_feat)
        return nn.Conv2d(n_feat, n_feat, 3, 1, 1)

    def switch_to_deploy(self, prune):
        n_feat, _, _, _ = self.body[0].conv.weight.data.shape

        self.body[0].switch_to_deploy()
        self.body[2].switch_to_deploy()
        if self.version >= 4:
            self.body[4].switch_to_deploy()

        if self.version == 1:
            body = self.body
            self.__delattr__("body")
            # redundancy
            delattr(self, "body")
            del self._modules["body"]
            del self._parameters["body"]
            self.body = nn.Sequential(
                nn.Conv2d(n_feat, n_feat, 3, 1, 1),
                nn.LeakyReLU(0.05, inplace=True),
                nn.Conv2d(n_feat, n_feat, 3, 1, 1),
            )
        elif self.version in [2, 3]:
            k3x3 = self.body[2].conv.weight.data
            b3x3 = self.body[2].conv.bias.data
            k1x1 = self.body[3].conv.weight.data
            b1x1 = self.body[3].conv.bias.data
            merge_w = F.conv2d(input=k3x3.permute(1, 0, 2, 3), weight=k1x1).permute(
                1, 0, 2, 3
            )
            merge_b = F.conv2d(
                input=b3x3.unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
                weight=k1x1,
                bias=b1x1,
            ).view(-1)
            self.body[2].conv.weight.data[0:1, :, ...] = merge_w.float()
            self.body[2].conv.bias.data[0:1] = merge_b.float()
            body = self.body
            self.__delattr__("body")
            # redundancy
            delattr(self, "body")
            del self._modules["body"]
            del self._parameters["body"]
        else:
            k3x3 = self.body[4].conv.weight.data
            b3x3 = self.body[4].conv.bias.data
            k1x1 = self.body[5].conv.weight.data
            b1x1 = self.body[5].conv.bias.data
            merge_w = F.conv2d(input=k3x3.permute(1, 0, 2, 3), weight=k1x1).permute(
                1, 0, 2, 3
            )
            merge_b = F.conv2d(
                input=b3x3.unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
                weight=k1x1,
                bias=b1x1,
            ).view(-1)
            self.body[4].conv.weight.data[0:1, :, ...] = merge_w.float()
            self.body[4].conv.bias.data[0:1] = merge_b.float()
            body = self.body
            self.__delattr__("body")
            # redundancy
            delattr(self, "body")
            del self._modules["body"]
            del self._parameters["body"]

        if self.version == 2:
            self.body = nn.Sequential(
                nn.Conv2d(n_feat, n_feat, 3, 1, 1),
                nn.LeakyReLU(0.05, inplace=True),
                nn.Conv2d(n_feat, n_feat, 3, 1, 1),
                LocalAttention(n_feat, self.f),
            )
        elif self.version == 3:
            self.body = nn.Sequential(
                nn.Conv2d(n_feat, n_feat, 3, 1, 1),
                nn.LeakyReLU(0.05, inplace=True),
                nn.Conv2d(n_feat, n_feat, 3, 1, 1),
                LocalAttention(n_feat, self.f, speed=True),
            )
        else:
            self.body = nn.Sequential(
                nn.Conv2d(n_feat, n_feat, 3, 1, 1),
                nn.LeakyReLU(0.05, inplace=True),
                nn.Conv2d(n_feat, n_feat, 3, 1, 1),
                nn.LeakyReLU(0.05, inplace=True),
                nn.Conv2d(n_feat, n_feat, 3, 1, 1),
                LocalAttention(n_feat, self.f),
            )

        self.body[0].weight.data = body[0].conv.weight.data
        self.body[0].bias.data = body[0].conv.bias.data
        self.body[2].weight.data = body[2].conv.weight.data
        self.body[2].bias.data = body[2].conv.bias.data

        if self.version >= 4:
            self.body[4].weight.data = body[4].conv.weight.data
            self.body[4].bias.data = body[4].conv.bias.data

        if self.version == 2:
            for i in [0, 2, 3]:
                self.body[3].body[i].weight.data = body[4].body[i].weight.data
                self.body[3].body[i].bias.data = body[4].body[i].bias.data
        elif self.version == 3:
            for i in [0]:
                self.body[3].body[i].weight.data = body[4].body[i].weight.data
                self.body[3].body[i].bias.data = body[4].body[i].bias.data
        else:
            for i in [0, 2, 3]:
                self.body[5].body[i].weight.data = body[6].body[i].weight.data
                self.body[5].body[i].bias.data = body[6].body[i].bias.data

        if prune:
            x = self.body[0].weight.data
            self.body[0].weight.data = torch.where(x.abs() < 1e-2, 0, x)
            x = self.body[2].weight.data
            self.body[2].weight.data = torch.where(x.abs() < 1e-2, 0, x)
            if self.version >= 4:
                x = self.body[4].weight.data
                self.body[4].weight.data = torch.where(x.abs() < 1e-2, 0, x)

        for para in self.parameters():
            para.detach_()


class PConv(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.n_feat = n_feat
        self.conv = nn.Conv2d(n_feat, 1, 1, 1, 0)
        self.init_weights()

    def forward(self, x):
        x1 = self.conv(x)
        x2 = x[:, 1:].clone()
        return torch.cat([x1, x2], dim=1)

    def init_weights(self):
        self.conv.weight.data.fill_(1 / self.n_feat)
        self.conv.bias.data.fill_(0)


class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(
            kernel_size, stride, padding, count_include_pad=False
        )

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool


class LocalAttention(nn.Module):
    """attention based on local importance"""

    def __init__(self, channels, f=16, speed=False):
        super().__init__()
        self.speed = speed

        if self.speed:
            self.body = nn.Sequential(
                # sample importance
                nn.Conv2d(channels, channels, 3, 1, 1),
                # to heatmap
                nn.Sigmoid(),
            )
        else:
            self.body = nn.Sequential(
                # sample importance
                nn.Conv2d(channels, f, 1),
                SoftPooling2D(7, stride=3),
                nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(f, channels, 3, padding=1),
                # to heatmap
                nn.Sigmoid(),
            )

        self.gate = nn.Sequential(nn.Sigmoid())

    def forward(self, x):
        """forward"""
        # interpolate the heat map
        g = self.gate(x[:, :1].clone())
        if self.speed:
            w = self.body(x)
        else:
            w = F.interpolate(
                self.body(x),
                (x.size(2), x.size(3)),
                mode="bilinear",
                align_corners=False,
            )
        return x * w * g


@ARCH_REGISTRY.register()
class plainusr(nn.Module):
    """PlainUSR: Chasing Faster ConvNet for Efficient Super-Resolution:
    https://arxiv.org/abs/2409.13435
    Code adapted from:
    https://github.com/icandle/PlainUSR
    """

    def __init__(self, n_feat=64, im_feat=(64, 48, 32), attn_feat=16, scale=upscale):
        super().__init__()
        self.n_feat = n_feat
        self.scale = scale
        self.im_feat = im_feat
        self.head = nn.Conv2d(3, n_feat + 3, 3, 1, 1)

        if self.n_feat <= 32:
            self.block1 = Block(im_feat[0], version=1, f=attn_feat)
            self.blockm = Block(im_feat[1], version=3, f=attn_feat)
            self.block2 = Block(im_feat[0], version=1, f=attn_feat)
        elif self.n_feat >= 80:
            self.block1 = Block(im_feat[0], version=4, f=attn_feat)
            self.block2 = Block(im_feat[1], version=4, f=attn_feat)
            self.blockm = Block(im_feat[2], version=4, f=attn_feat)
            self.block3 = Block(im_feat[1], version=4, f=attn_feat)
            self.block4 = Block(im_feat[0], version=4, f=attn_feat)
        else:
            self.block1 = Block(im_feat[0], version=2, f=attn_feat)
            self.block2 = Block(im_feat[1], version=2, f=attn_feat)
            self.blockm = Block(im_feat[2], version=2, f=attn_feat)
            self.block3 = Block(im_feat[1], version=2, f=attn_feat)
            self.block4 = Block(im_feat[0], version=2, f=attn_feat)

        self.tail = nn.Sequential(
            nn.Conv2d(n_feat + 3, 3 * (scale**2), 3, 1, 1), nn.PixelShuffle(scale)
        )

        self.init_weights()

    def init_weights(self):
        scale_squared = self.scale**2
        self.head.weight.data[-3, 0, 1, 1] += 1
        self.head.weight.data[-2, 1, 1, 1] += 1
        self.head.weight.data[-1, 2, 1, 1] += 1

        self.tail[0].weight.data[:scale_squared, -3, 1, 1] += 1
        self.tail[0].weight.data[scale_squared : 2 * scale_squared, -2, 1, 1] += 1
        self.tail[0].weight.data[2 * scale_squared :, -1, 1, 1] += 1

    def fast_forward(self, x):
        x = self.head(x)

        if self.n_feat <= 32:
            x[:, : self.im_feat[0]] = self.block1(x[:, : self.im_feat[0]])
            x[:, : self.im_feat[1]] = self.blockm(x[:, : self.im_feat[1]])
            x[:, : self.im_feat[0]] = self.block2(x[:, : self.im_feat[0]])
        else:
            x[:, : self.im_feat[0]] = self.block1(x[:, : self.im_feat[0]])
            x[:, : self.im_feat[1]] = self.block2(x[:, : self.im_feat[1]])
            x[:, : self.im_feat[2]] = self.blockm(x[:, : self.im_feat[2]])
            x[:, : self.im_feat[1]] = self.block3(x[:, : self.im_feat[1]])
            x[:, : self.im_feat[0]] = self.block4(x[:, : self.im_feat[0]])

        return self.tail(x)

    def forward(self, x):
        if not self.training:
            return self.fast_forward(x)

        x = self.head(x)

        if self.n_feat <= 32:
            x, pic = x.split([self.im_feat[0], 3], 1)
            x = self.block1(x)
            x1, x2 = x.split([self.im_feat[1], self.im_feat[0] - self.im_feat[1]], 1)
            x1 = self.blockm(x1)
            x1 = torch.cat([x1, x2], 1)
            x1 = self.block2(x1)
        else:
            x, pic = x.split([self.im_feat[0], 3], 1)
            x = self.block1(x)
            x1, x2 = x.split([self.im_feat[1], self.im_feat[0] - self.im_feat[1]], 1)
            x1 = self.block2(x1)
            x11, x12 = x1.split([self.im_feat[2], self.im_feat[1] - self.im_feat[2]], 1)
            x11 = self.blockm(x11)
            x1 = torch.cat([x11, x12], 1)
            x1 = self.block3(x1)
            x = torch.cat([x1, x2], 1)
            x = self.block4(x)

        x = torch.cat([x, pic], 1)
        return self.tail(x)


@ARCH_REGISTRY.register()
def plainusr_ultra(**kwargs):
    return plainusr(n_feat=32, im_feat=(32, 16), attn_feat=4, **kwargs)


@ARCH_REGISTRY.register()
def plainusr_large(**kwargs):
    return plainusr(n_feat=80, im_feat=(80, 64, 48), attn_feat=16, **kwargs)
