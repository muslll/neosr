import torch
import torch.nn.modules.module
from torch import Tensor, nn
from torch.nn import Conv2d, Module, Parameter
from torch.nn.utils import spectral_norm
from torchvision import models
from torchvision.models import ResNet18_Weights

from neosr.archs.arch_util import DySample
from neosr.utils.registry import ARCH_REGISTRY


def conv3otherMish(
    in_planes: int,
    out_planes: int,
    kernel_size: int | None = None,
    stride: int | None = None,
    padding: int | None = None,
) -> nn.Sequential:
    # 3x3 convolution with padding and mish
    if kernel_size is None:
        kernel_size = 3
    assert isinstance(kernel_size, int | tuple), "kernel_size is not in (int, tuple)!"

    if stride is None:
        stride = 1
    assert isinstance(stride, int | tuple), "stride is not in (int, tuple)!"

    if padding is None:
        padding = 1
    assert isinstance(padding, int | tuple), "padding is not in (int, tuple)!"

    return nn.Sequential(
        spectral_norm(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            )
        ),
        nn.Mish(inplace=True),
    )


def l2_norm(x: Tensor) -> Tensor:
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


class ConvBnMish(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        ksize: int,
        stride: int,
        pad: int,
        dilation: int = 1,
        groups: int = 1,
        has_bn: bool = True,
        norm_layer: type[nn.BatchNorm2d] = nn.BatchNorm2d,
        bn_eps: float = 1e-5,
        has_mish: bool = True,
        inplace: bool = True,
        has_bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv = spectral_norm(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                dilation=dilation,
                groups=groups,
                bias=has_bias,
            )
        )
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_mish = has_mish
        if self.has_mish:
            self.mish = nn.Mish(inplace=inplace)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_mish:
            x = self.mish(x)
        return x


class Attention(Module):
    def __init__(self, in_places: int, scale: int = 8, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = Conv2d(
            in_channels=in_places, out_channels=in_places // scale, kernel_size=1
        )
        self.key_conv = Conv2d(
            in_channels=in_places, out_channels=in_places // scale, kernel_size=1
        )
        self.value_conv = Conv2d(
            in_channels=in_places, out_channels=in_places, kernel_size=1
        )

    def forward(self, x: Tensor) -> Tensor:
        # Apply the feature map to the queries and keys
        batch_size, chnnels, height, width = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)

        tailor_sum = 1 / (
            width * height
            + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps)
        )
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)

        matrix = torch.einsum("bmn, bcn->bmc", K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (self.gamma * weight_value).contiguous()


class AttentionAggregationModule(nn.Module):
    def __init__(self, in_chan: int, out_chan: int) -> None:
        super().__init__()
        self.convblk = ConvBnMish(in_chan, out_chan, ksize=1, stride=1, pad=0)
        self.conv_atten = Attention(out_chan)

    def forward(self, s5: Tensor, s4: Tensor, s3: Tensor, s2: Tensor) -> Tensor:
        fcat = torch.cat([s5, s4, s3, s2], dim=1)
        feat = self.convblk(fcat)
        atten = self.conv_atten(feat)
        return atten + feat


class Conv3x3GNMish(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, upsample: bool = False
    ) -> None:
        super().__init__()
        self.upsample = upsample
        self.dysample = DySample(
            in_channels=64, out_ch=64, scale=2, groups=4, end_convolution=True
        )
        self.block = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
                )
            ),
            nn.GroupNorm(32, out_channels),
            nn.Mish(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        if self.upsample:
            x = self.dysample(x)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels: int, skip_channels: int) -> None:
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)
        self.dysample = DySample(
            in_channels=64, out_ch=64, scale=2, groups=4, end_convolution=False
        )

    def forward(self, x: Tensor) -> Tensor:
        x, skip = x
        x = self.dysample(x)
        skip = self.skip_conv(skip)
        return x + skip


class SegmentationBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, n_upsamples: int = 0
    ) -> None:
        super().__init__()

        blocks = [Conv3x3GNMish(in_channels, out_channels, upsample=bool(n_upsamples))]
        if n_upsamples > 1:
            blocks.extend(
                Conv3x3GNMish(out_channels, out_channels, upsample=True)
                for _ in range(1, n_upsamples)
            )
        self.block = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


@ARCH_REGISTRY.register()
class ea2fpn(nn.Module):
    """Modified for the neosr project, based on 'A2-FPN for Semantic
    Segmentation of Fine-Resolution Remotely Sensed Images':
    https://doi.org/10.1080/01431161.2022.2030071.
    """

    def __init__(
        self,
        class_num: int = 6,
        encoder_channels: tuple[int, int, int, int] = (512, 256, 128, 64),
        pyramid_channels: int = 64,
        segmentation_channels: int = 64,
        dropout: float = 0.2,
        **kwargs,  # noqa: ARG002
    ) -> None:
        super().__init__()
        self.base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base_layers = list(self.base_model.children())
        # ==> encoder layers
        self.layer_down0 = nn.Sequential(
            *self.base_layers[:3]
        )  # size=(N, 64, x.H/2, x.W/2)
        self.layer_down1 = nn.Sequential(
            *self.base_layers[3:5]
        )  # size=(N, 64, x.H/4, x.W/4)
        self.layer_down2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer_down3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer_down4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

        self.conv1 = spectral_norm(
            nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=(1, 1))
        )

        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.s5 = SegmentationBlock(
            pyramid_channels, segmentation_channels, n_upsamples=3
        )
        self.s4 = SegmentationBlock(
            pyramid_channels, segmentation_channels, n_upsamples=2
        )
        self.s3 = SegmentationBlock(
            pyramid_channels, segmentation_channels, n_upsamples=1
        )
        self.s2 = SegmentationBlock(
            pyramid_channels, segmentation_channels, n_upsamples=0
        )

        self.attention = AttentionAggregationModule(
            segmentation_channels * 4, segmentation_channels * 4
        )
        self.final_conv = spectral_norm(
            nn.Conv2d(segmentation_channels * 4, class_num, kernel_size=1, padding=0)
        )
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.dysample = DySample(
            in_channels=6, out_ch=3, scale=4, groups=3, end_convolution=False
        )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(
            m, nn.LayerNorm | nn.BatchNorm2d | nn.GroupNorm | nn.InstanceNorm2d
        ):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        # ==> get encoder features
        c1 = self.layer_down0(x)
        c2 = self.layer_down1(c1)
        c3 = self.layer_down2(c2)
        c4 = self.layer_down3(c3)
        c5 = self.layer_down4(c4)
        # c5, c4, c3, c2, _ = x

        p5 = self.conv1(c5)
        p4 = self.p4([p5, c4])
        p3 = self.p3([p4, c3])
        p2 = self.p2([p3, c2])

        s5 = self.s5(p5)
        s4 = self.s4(p4)
        s3 = self.s3(p3)
        s2 = self.s2(p2)

        out = self.dropout(self.attention(s5, s4, s3, s2))
        out = self.final_conv(out)
        return self.dysample(out)
