from torch.nn import functional as F
from torch import nn

from neosr.archs.arch_util import net_opt
from neosr.utils.registry import ARCH_REGISTRY

upscale, __ = net_opt()


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    """Adapted from timm.
    SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """

    def __init__(
        self,
        channels,
        rd_ratio=1.0 / 16,
        rd_channels=None,
        rd_divisor=8,
        add_maxpool=False,
        bias=True,
        act_layer=nn.ReLU,
        norm_layer=None,
        gate_layer="sigmoid",
    ):
        super(SqueezeExcite, self).__init__()
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=bias)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=bias)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


def pad_tensor(t, pattern):
    pattern = pattern.view(1, -1, 1, 1)
    t = F.pad(t, (1, 1, 1, 1), "constant", 0)
    t[:, :, 0:1, :] = pattern
    t[:, :, -1:, :] = pattern
    t[:, :, :, 0:1] = pattern
    t[:, :, :, -1:] = pattern

    return t


class RepConv(nn.Module):
    """Residual in residual reparameterizable block.
    Using reparameterizable block to replace single 3x3 convolution.

    Args:
        n_feats (int): The number of feature maps.
        ratio (int): Expand ratio.

    NOTE: inference code uses only self.res_conv3x3 on x. However,
    it requires reparametrization, so nn.Module self.training can't
    be used, otherwise .eval() will produce wrong results. Script:
    https://github.com/huai-chang/RVSR/blob/main/reparameterize.py
    """

    def __init__(self, n_feats, ratio=2):
        super(RepConv, self).__init__()
        self.expand_conv = nn.Conv2d(n_feats, n_feats * ratio, 1, 1, 0)
        self.fea_conv = nn.Conv2d(n_feats * ratio, n_feats * ratio, 3, 1, 0)
        self.reduce_conv = nn.Conv2d(n_feats * ratio, n_feats, 1, 1, 0)

        self.expand1_conv = nn.Conv2d(n_feats, n_feats * ratio, 1, 1, 0)
        self.fea1_conv = nn.Conv2d(n_feats * ratio, n_feats * ratio, 3, 1, 0)
        self.reduce1_conv = nn.Conv2d(n_feats * ratio, n_feats, 1, 1, 0)

        self.res_conv3x3 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.res_conv1x1 = nn.Conv2d(n_feats, n_feats, 1)


    def forward(self, x):
        res3 = self.res_conv3x3(x)
        res1 = self.res_conv1x1(x)
        res = x

        branch1 = self.expand_conv(x)
        b0 = self.expand_conv.bias
        branch1 = pad_tensor(branch1, b0)
        branch1 = self.fea_conv(branch1)
        branch1 = self.reduce_conv(branch1)

        branch2 = self.expand1_conv(x)
        b0 = self.expand1_conv.bias
        branch2 = pad_tensor(branch2, b0)
        branch2 = self.fea1_conv(branch2)
        branch2 = self.reduce1_conv(branch2)
        out = branch1 + branch2 + res + res1 + res3

        return out


class FFN(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv2d(N, N * 2, 1), nn.GELU(), nn.Conv2d(N * 2, N, 1)
        )

    def forward(self, x):
        return self.ffn(x) + x


class RepViT(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.token_mixer = RepConv(N)
        self.channel_mixer = FFN(N)
        self.attn = SqueezeExcite(N, 0.25)

    def forward(self, x):
        return self.channel_mixer(self.attn(self.token_mixer(x)))


@ARCH_REGISTRY.register()
class rvsr(nn.Module):
    def __init__(self, scale=upscale, N=16):
        super(rvsr, self).__init__()
        self.scale = scale
        self.head = nn.Sequential(nn.Conv2d(3, N, 3, 1, 1))
        self.body = nn.Sequential(
            RepViT(N),
            RepViT(N),
            RepViT(N),
            RepViT(N),
            RepViT(N),
            RepViT(N),
            RepViT(N),
            RepViT(N),
        )

        self.tail = nn.Sequential(
            RepConv(N), nn.Conv2d(N, 3 * scale * scale, 1), nn.PixelShuffle(4)
        )

    def forward(self, x):
        head = self.head(x)
        body = self.body(head) + head
        h = self.tail(body)
        base = F.interpolate(
            x, scale_factor=self.scale, mode="bilinear", align_corners=False
        )
        out = h + base

        return out
