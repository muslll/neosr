import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from neosr.archs.arch_util import net_opt, to_2tuple
from neosr.utils.registry import ARCH_REGISTRY

upscale, __ = net_opt()


# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        if self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight[:, None, None] * x + self.bias[:, None, None]
        return None


# SE
class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


# Channel MLP: Conv1*1 -> Conv1*1
class ChannelMLP(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.mlp(x)


# MBConv: Conv1*1 -> DW Conv3*3 -> [SE] -> Conv1*1
class MBConv(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.mbconv = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim),
            nn.GELU(),
            SqueezeExcitation(hidden_dim),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.mbconv(x)


# CCM
class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.ccm(x)


class ConvPosEnc(nn.Module):
    """Depth-wise convolution to get the positional information."""

    def __init__(self, dim, k=3):
        super().__init__()
        self.proj = nn.Conv2d(
            dim, dim, to_2tuple(k), to_2tuple(1), to_2tuple(k // 2), groups=dim
        )

    def forward(self, x, size: tuple[int, int]):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(feat)
        feat = feat.flatten(2).transpose(1, 2)
        return x + feat


##############################################################
# Downsample ViT
class downsample_vit(nn.Module):
    def __init__(self, dim, window_size=8, attn_drop=0.0, proj_drop=0.0, down_scale=2):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.scale = dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        return (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, window_size, window_size, C)
        )

    def window_reverse(self, windows, window_size, h, w):
        """
        Args:
            windows: (num_windows*b, window_size, window_size, c)
            window_size (int): Window size
            h (int): Height of image
            w (int): Width of image

        Returns:
            x: (b, h, w, c)
        """
        b = int(windows.shape[0] / (h * w / window_size / window_size))
        x = windows.view(
            b, h // window_size, w // window_size, window_size, window_size, -1
        )
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.window_size, self.window_size
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = (
            x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)
        )  # B', C, H', W'

        lepe = func(x)  # B', C, H', W'
        lepe = lepe.reshape(-1, C, H_sp * W_sp).permute(0, 2, 1).contiguous()

        x = x.reshape(-1, C, H_sp * W_sp).permute(0, 2, 1).contiguous()
        return x, lepe

    def forward(self, x):
        _B, C, H, W = x.shape

        ################################
        # 1. window partition
        ################################
        x = x.permute(0, 2, 3, 1)
        x_window = self.window_partition(x, self.window_size).permute(0, 3, 1, 2)
        x_window = x_window.permute(0, 2, 3, 1).view(
            -1, self.window_size * self.window_size, C
        )

        ################################
        # 2. make qkv
        ################################
        qkv = self.qkv(x_window)
        # qkv = qkv.permute(0,2,3,1)
        # qkv = qkv.reshape(-1, self.window_size * self.window_size, 3*C)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        ################################
        # 3. attn and PE
        ################################
        v, lepe = self.get_lepe(v, self.get_v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        # x = x.reshape(-1, self.window_size, self.window_size, C)
        # x = x.permute(0,3,1,2)

        ################################
        # 4. proj and drop
        ################################
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.reshape(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(x, self.window_size, H, W)

        return x.permute(0, 3, 1, 2)


##############################################################
# LHSB - split dim and define 4 attn blocks
class LHSB(nn.Module):
    def __init__(self, dim, attn_drop=0.0, proj_drop=0.0, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList([
            downsample_vit(
                dim // 4,
                window_size=8,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                down_scale=2**i,
            )
            for i in range(self.n_levels)
        ])

        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []

        downsampled_feat = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2**i, w // 2**i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                downsampled_feat.append(s)

            else:
                downsampled_feat.append(xc[i])

        for i in reversed(range(self.n_levels)):
            s = self.mfr[i](downsampled_feat[i])
            s_upsample = F.interpolate(
                s, size=(s.shape[2] * 2, s.shape[3] * 2), mode="nearest"
            )

            if i > 0:
                downsampled_feat[i - 1] = downsampled_feat[i - 1] + s_upsample

            s_original_shape = F.interpolate(s, size=(h, w), mode="nearest")
            out.append(s_original_shape)

        out = self.aggr(torch.cat(out, dim=1))
        return self.act(out) * x


##############################################################
# Block
class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        # Multiscale Block
        self.lhsb = LHSB(dim, attn_drop=attn_drop, proj_drop=drop)

        # Feedforward layer
        self.ccm = CCM(dim, ffn_scale)

    def forward(self, x):
        x = self.lhsb(self.norm1(x)) + x
        return self.ccm(self.norm2(x)) + x


@ARCH_REGISTRY.register()
class lmlt(nn.Module):
    def __init__(
        self,
        dim=60,
        n_blocks=8,
        ffn_scale=2.0,
        upscaling_factor=upscale,
        window_size=8,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)
        self.window_size = window_size

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)
        ]  # stochastic depth decay rule

        self.feats = nn.Sequential(*[
            AttBlock(
                dim,
                ffn_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
            )
            for i in range(n_blocks)
        ])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor),
        )

    def check_img_size(self, x):
        _, _, h, w = x.size()
        downsample_scale = 8
        scaled_size = self.window_size * downsample_scale

        mod_pad_h = (scaled_size - h % scaled_size) % scaled_size
        mod_pad_w = (scaled_size - w % scaled_size) % scaled_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward(self, x):
        _B, _C, H, W = x.shape

        # check image size
        x = self.check_img_size(x)

        # patch embed
        x = self.to_feat(x)

        # module, and return to original shape
        x = self.feats(x) + x
        x = x[:, :, :H, :W]

        # reconstruction
        return self.to_img(x)


@ARCH_REGISTRY.register()
def lmlt_tiny(**kwargs):
    return lmlt(dim=36, **kwargs)


@ARCH_REGISTRY.register()
def lmlt_large(**kwargs):
    return lmlt(dim=84, **kwargs)
