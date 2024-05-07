import numbers

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from neosr.utils.registry import ARCH_REGISTRY

from .arch_util import net_opt

upscale, training = net_opt()


class ElementScale(nn.Module):
    def __init__(self, embed_dims, init_value=0.0, requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class RCCA(nn.Module):
    def __init__(self, embed_dims, feedforward_channels, kernel_size=3):
        super(RCCA, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels,
        )

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels, out_channels=1, kernel_size=1
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True
        )

        self.cca = CCALayer(self.feedforward_channels, self.feedforward_channels // 4)

        self.act = nn.GELU()
        self.decompose_act = nn.GELU()

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.act(x)
        x1 = self.feat_decompose(x)
        x2 = self.cca(x)
        x = x1 + x2
        return x + input


class BSConvU(nn.Module):
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
    ):
        super().__init__()

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
        self.dw = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode="reflect",
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


def stdv_channels(F):
    assert F.dim() == 4
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (
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


def get_local_weights(residual, ksize, padding):
    pad = padding
    residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode="reflect")
    unfolded_residual = residual_pad.unfold(2, ksize, 3).unfold(3, ksize, 3)
    pixel_level_weight = (
        torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True)
        .squeeze(-1)
        .squeeze(-1)
    )

    return pixel_level_weight


class ESA(nn.Module):
    def __init__(self, num_feat=24, conv=nn.Conv2d):
        super(ESA, self).__init__()
        f = num_feat // 4
        self.conv1 = nn.Conv2d(num_feat, f, 1)
        self.conv_f = nn.Conv2d(f, f, 1)
        self.conv2_1 = conv(f, f, 3, 2, 1, padding_mode="reflect")
        self.conv2_2 = conv(f, f, 3, 2, 1, padding_mode="reflect")
        self.conv2_3 = conv(f, f, 3, 2, 1, padding_mode="reflect")
        self.conv2_4 = conv(f, f, 3, 2, 1, padding_mode="reflect")
        self.maxPooling_1 = nn.MaxPool2d(kernel_size=5, stride=3, padding=1)
        self.maxPooling_2 = nn.MaxPool2d(kernel_size=7, stride=3, padding=1)
        self.conv_max_1 = conv(f, f, kernel_size=3)
        self.conv_max_2 = conv(f, f, kernel_size=3)
        self.var_3 = get_local_weights
        self.var_4 = get_local_weights

        self.conv3_1 = conv(f, f, kernel_size=3)
        self.conv3_2 = conv(f, f, kernel_size=3)
        self.conv4 = nn.Conv2d(f, num_feat, 1)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, input):
        c1_ = self.conv1(input)  # channel squeeze
        c1_1 = self.maxPooling_1(self.conv2_1(c1_))  # strided conv 5
        c1_2 = self.maxPooling_2(self.conv2_2(c1_))  # strided conv 7
        c1_3 = self.var_3(self.conv2_3(c1_), 7, padding=1)  # strided local-var 7
        c1_4 = self.var_4(self.conv2_4(c1_), 5, padding=1)  # strided local-var 5

        v_range_1 = self.conv3_1(self.GELU(self.conv_max_1(c1_1 + c1_4)))
        v_range_2 = self.conv3_2(self.GELU(self.conv_max_2(c1_2 + c1_3)))

        c3_1 = F.interpolate(
            v_range_1,
            (input.size(2), input.size(3)),
            mode="bilinear",
            align_corners=False,
        )
        c3_2 = F.interpolate(
            v_range_2,
            (input.size(2), input.size(3)),
            mode="bilinear",
            align_corners=False,
        )

        cf = self.conv_f(c1_)
        c4 = self.conv4(c3_1 + c3_2 + cf)
        m = self.sigmoid(c4)

        return input * m


class MVSA(nn.Module):
    def __init__(self, c_dim, conv):
        super().__init__()
        self.body = nn.Sequential(ESA(c_dim, conv))

    def forward(self, x):
        sa_x = self.body(x)
        sa_x += x
        return sa_x


# Layer Norm
def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    """
    GDFN in Restormer: [github] https://github.com/swz30/Restormer
    """

    def __init__(self, dim, ffn_expansion_factor, bias, input_resolution=None):
        super(FeedForward, self).__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.ffn_expansion_factor = ffn_expansion_factor

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class SparseAttention(nn.Module):
    """
    SparseGSA is based on MDTA
    MDTA in Restormer: [github] https://github.com/swz30/Restormer
    TLC: [github] https://github.com/megvii-research/TLC
    We use TLC-Restormer in forward function and only use it in test mode
    """

    def __init__(
        self,
        dim,
        num_heads,
        bias,
        tlc_flag=True,
        tlc_kernel=48,
        activation="relu",
        input_resolution=None,
    ):
        super(SparseAttention, self).__init__()
        self.training = training
        self.tlc_flag = tlc_flag  # TLC flag for validation and test

        self.dim = dim
        self.input_resolution = input_resolution

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.act = nn.Identity()

        # ['gelu', 'sigmoid'] is for ablation study
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()

        # [x2, x3, x4] -> [96, 72, 48]
        self.kernel_size = [tlc_kernel, tlc_kernel]

    def _forward(self, qkv):
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        # attn = attn.softmax(dim=-1)
        attn = self.act(attn)  # Sparse Attention due to ReLU's property

        out = attn @ v

        return out

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))

        if self.training or not self.tlc_flag:
            out = self._forward(qkv)
            out = rearrange(
                out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
            )

            out = self.project_out(out)
            return out

        # Then we use the TLC methods in test mode
        qkv = self.grids(qkv)  # convert to local windows
        out = self._forward(qkv)
        out = rearrange(
            out,
            "b head c (h w) -> b (head c) h w",
            head=self.num_heads,
            h=qkv.shape[-2],
            w=qkv.shape[-1],
        )
        out = self.grids_inverse(out)  # reverse

        out = self.project_out(out)
        return out

    # Code from [megvii-research/TLC] https://github.com/megvii-research/TLC
    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c // 3, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math

        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i : i + k1, j : j + k2])
                idxes.append({"i": i, "j": j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx["i"]
            j = each_idx["j"]
            preds[0, :, i : i + k1, j : j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i : i + k1, j : j + k2] += 1.0

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt


class SparseAttentionLayerBlock(nn.Module):
    def __init__(
        self,
        dim,
        restormer_num_heads=3,
        restormer_ffn_expansion_factor=2.0,
        tlc_flag=True,
        tlc_kernel=48,
        activation="relu",
        input_resolution=None,
    ):
        super(SparseAttentionLayerBlock, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.norm3 = LayerNorm(dim, LayerNorm_type="WithBias")
        self.norm4 = LayerNorm(dim, LayerNorm_type="WithBias")

        # We use SparseGSA inplace MDTA
        self.restormer_attn = SparseAttention(
            dim,
            num_heads=restormer_num_heads,
            bias=False,
            tlc_flag=tlc_flag,
            tlc_kernel=tlc_kernel,
            activation=activation,
            input_resolution=input_resolution,
        )
        self.restormer_ffn = FeedForward(
            dim,
            ffn_expansion_factor=restormer_ffn_expansion_factor,
            bias=False,
            input_resolution=input_resolution,
        )

    def forward(self, x):
        x = self.restormer_attn(self.norm3(x)) + x
        x = self.restormer_ffn(self.norm4(x)) + x
        return x


class AGDB(nn.Module):
    def __init__(self, in_channels, conv=nn.Conv2d):
        super(AGDB, self).__init__()
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = conv(in_channels, self.rc, 3, 1, 1)
        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c2_r = conv(self.remaining_channels, self.rc, 3, 1, 1)
        self.c3_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c3_r = conv(self.remaining_channels, self.rc, 3, 1, 1)

        self.c4 = nn.Conv2d(self.remaining_channels, self.dc, 3, 1, 1, groups=self.dc)
        self.act = nn.GELU()

        self.c5 = nn.Conv2d(self.dc * 4, in_channels, 1)
        self.esa = MVSA(in_channels, conv)
        self.cca = RCCA(in_channels, int(in_channels * 1))
        self.ea = SparseAttentionLayerBlock(in_channels)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = self.c1_r(input)
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.c2_r(r_c1)
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.c3_r(r_c2)
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)

        out_fused = self.esa(out)
        out_fused = self.cca(out_fused)
        out_fused = self.ea(out_fused)

        return out_fused + input


def UpsampleOneStep(in_channels, out_channels, upscale_factor=4):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])


@ARCH_REGISTRY.register()
class agdn(nn.Module):
    def __init__(
        self,
        num_in_ch=3,
        num_feat=42,
        num_block=7,
        num_out_ch=3,
        upscale=upscale,
        light=False,
        rgb_mean=(0.5, 0.5, 0.5),
        **kwargs,
    ):
        super(agdn, self).__init__()
        self.conv = BSConvU
        self.scale = upscale
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.light = light

        self.fea_conv = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.B1 = AGDB(in_channels=num_feat, conv=self.conv)
        self.B2 = AGDB(in_channels=num_feat, conv=self.conv)
        self.B3 = AGDB(in_channels=num_feat, conv=self.conv)
        self.B4 = AGDB(in_channels=num_feat, conv=self.conv)
        if not self.light:
            self.B5 = AGDB(in_channels=num_feat, conv=self.conv)
            self.B6 = AGDB(in_channels=num_feat, conv=self.conv)
            self.B7 = AGDB(in_channels=num_feat, conv=self.conv)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        self.GELU = nn.GELU()
        self.c2 = self.conv(num_feat, num_feat, 3, 1, 1)

        self.upsampler = UpsampleOneStep(num_feat, num_out_ch, upscale_factor=upscale)

    def forward(self, input):
        self.mean = self.mean.type_as(input)
        input = input - self.mean

        out_fea = self.fea_conv(input)

        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        if not self.light:
            out_B5 = self.B5(out_B4)
            out_B6 = self.B6(out_B5)
            out_B7 = self.B7(out_B6)

        if not self.light:
            cat_out = [out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7]
        else:
            cat_out = [out_B1, out_B2, out_B3, out_B4]

        out = (
            self.upsampler(
                self.c2(self.GELU(self.c1(torch.cat(cat_out, dim=1)))) + out_fea
            )
            + self.mean
        )

        return out


@ARCH_REGISTRY.register()
def agdn_s(**kwargs):
    return agdn(num_feat=24, num_block=4, light=True, **kwargs)
