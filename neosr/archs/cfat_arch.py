import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.init import trunc_normal_

from neosr.utils.registry import ARCH_REGISTRY
from .arch_util import DropPath, net_opt, to_2tuple

upscale, training = net_opt()


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x):  # [1, 180, 64, 64]
        return self.cab(x)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention_D(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()

        # Arguments
        self.dim = dim  # 180
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads  # 6
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # Module
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, rpi, mask=None):  # [16, 16*16, 180]
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape  # [16, 256, 180]

        qkv = (
            self.qkv(x)
            .reshape(b_, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # [16, 256, 540]->[3, 16, 6, 256, 30]
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple) #[16, 6, 256, 30]

        q = q * self.scale  # scale=0.18257418583505536 #[16, 6, 256, 30]
        attn = q @ k.transpose(-2, -1)  # [16, 6, 256, 256]

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        # Wh*Ww,Wh*Ww,nH #[256, 256, 6]
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww #[6, 256, 256]
        attn = attn + relative_position_bias.unsqueeze(
            0
        )  # [16, 6, 256, 256]+[1, 6, 256, 256]=[16, 6, 256, 256]

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(
                1
            ).unsqueeze(0)  # [32, 6, 256, 256] [16, 256, 256]
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)  # [16, 6, 256, 256]

        x = (
            (attn @ v).transpose(1, 2).reshape(b_, n, c)
        )  # [16, 6, 256, 256]*[16, 6, 256, 30]=[16, 6, 256, 30]->[16, 256, 180]
        x = self.proj(x)  # [16, 256, 180]

        x = self.proj_drop(x)  # [16, 256, 180]

        return x


class WindowAttention_S(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()

        self.dim = dim  # 180
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads  # 6
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, rpi, mask=None, sp_mask=None):  # [16, 16*16, 180]
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape  # [16, 256, 180]

        qkv = (
            self.qkv(x)
            .reshape(b_, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # [16, 256, 540]->[3, 16, 6, 256, 30]
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple) #[16, 6, 256, 30]

        q = q * self.scale  # scale=0.18257418583505536 #[16, 6, 256, 30]
        attn = q @ k.transpose(-2, -1)  # [16, 6, 256, 256]

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        # Wh*Ww,Wh*Ww,nH #[256, 256, 6]
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww #[6, 256, 256]
        attn = attn + relative_position_bias.unsqueeze(
            0
        )  # [16, 6, 256, 256]+[1, 6, 256, 256]=[16, 6, 256, 256]

        if sp_mask is not None:
            nP = sp_mask.shape[0]
            attn = attn.view(b_ // nP, nP, self.num_heads, n, n) + sp_mask.unsqueeze(
                1
            ).unsqueeze(0)  # (B, nP, nHead, N, N)
            attn = attn.view(-1, self.num_heads, n, n)
            if mask is not None:
                nw = mask.shape[0]
                attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(
                    1
                ).unsqueeze(0)  # [32, 6, 256, 256] [16, 256, 256]
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)  # [16, 6, 256, 256]

        x = (
            (attn @ v).transpose(1, 2).reshape(b_, n, c)
        )  # [16, 6, 256, 256]*[16, 6, 256, 30]=[16, 6, 256, 30]->[16, 256, 180]
        x = self.proj(x)  # [16, 256, 180]

        x = self.proj_drop(x)  # [16, 256, 180]

        return x


def window_partition(x, window_size):  # [1, 64, 64, 180]
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(
        b, h // window_size, window_size, w // window_size, window_size, c
    )  # [1, 64//16, 16, 64//16, 16, 180]
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    )  # [1, 64//16, 64//16, 16, 16, 180]->#[16, 16, 16, 180]
    return windows


def window_reverse(windows, window_size, h, w):
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
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def window_partition_triangular(x, window_size, masks):  # [1, 64, 64, 180]
    b, h, w, c = x.shape
    m = len(masks)
    ws = window_size
    h_ws = h // ws
    w_ws = w // ws
    x = x.view(b, h_ws, ws, w_ws, ws, c)  # b, h/ws, ws, w/ws, ws, c
    windows = (
        x.permute(0, 1, 3, 5, 2, 4).contiguous().view(-1, ws, ws)
    )  # b, h/ws, w/ws, c, ws, ws-->b*(h_ws)*(w_ws)*c, ws, ws
    # window_mask=torch.zeros((len(masks), windows.shape[0], ws//2 * ws//2), dtype=windows.dtype).to(x.device)
    window_masks = []
    for mask in masks:
        mask = mask.expand(windows.shape[0], -1, -1)
        window_mask = windows[mask]
        window_masks.append(window_mask.unsqueeze(0))
    window_masks = torch.cat(window_masks, dim=0)
    window_masks = window_masks.view(m, windows.shape[0], -1)
    m, _, n = window_masks.shape
    window_masks = (
        window_masks.view(m, -1, c, n).permute(1, 0, 3, 2).contiguous()
    )  # [m, b*(h_ws)*(w_ws)*c, n]->[b*(h_ws)*(w_ws), m, n, c]
    return window_masks


def window_reverse_triangular(x, window_size, masks):
    b_, m, n, c = x.shape  # [b*(h_ws)*(w_ws), m, n, c]
    x = x.permute(1, 0, 3, 2).contiguous().view(m, -1)  # [m, b*(h_ws)*(w_ws)*c, n]
    reconstructed = torch.zeros((b_ * c, window_size, window_size), dtype=x.dtype).to(
        x.device
    )
    for mask, x_ in zip(masks, x, strict=False):
        mask = mask.expand(b_ * c, -1, -1)
        reconstructed[mask] = x_  # [b*(h_ws)*(w_ws)*c, ws, ws]
    return reconstructed


class HAB_D(nn.Module):
    r"""Hybrid Attention Block_Dense.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=16,
        shift_size=0,
        interval=0,
        triangular_flag=0,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.interval = interval
        self.mlp_ratio = mlp_ratio
        self.triangular_flag = triangular_flag
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        # assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention_D(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.conv_scale = conv_scale
        self.conv_block = CAB(
            num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self, x, x_size, rpi_sa, attn_mask, triangular_masks
    ):  # [1, 4096, 180], (64, 64)
        h, w = x_size
        b, _, c = x.shape
        # assert seq_len == h * w, "input feature has wrong size"
        shortcut = x

        x = self.norm1(x)
        x = x.view(b, h, w, c)  # [1, 64, 64, 180]
        conv_x = self.conv_block(x.permute(0, 3, 1, 2))  # [1, 180, 64, 64]
        conv_x = (
            conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        )  # [1, 4096, 180]
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            if self.shift_size == 8:
                attn_mask = attn_mask[0]
            if self.shift_size == 16:
                attn_mask = attn_mask[1]
            if self.shift_size == 24:
                attn_mask = attn_mask[2]
        else:
            shifted_x = x  # [1, 64, 64, 180]
            attn_mask = None
        if not self.triangular_flag:
            x_windows = window_partition(
                shifted_x, self.window_size
            )  # nw*b, window_size, window_size, c #[1, 64, 64, 180]->[16, 16, 16, 180]
            x_windows = x_windows.view(
                -1, self.window_size * self.window_size, c
            )  # nw*b, window_size*window_size, c  #[16, 16*16, 180]
        if self.triangular_flag:
            x_windows = window_partition_triangular(
                shifted_x, 2 * self.window_size, triangular_masks
            )  # nw*b, window_size, window_size, c #[1, 64, 64, 180]->[16, 16, 16, 180]
            _, m, n, _ = x_windows.shape  # [b*(h_ws)*(w_ws), m, n, c]
            x_windows = x_windows.view(
                -1, n, c
            )  # [b*(h_ws)*(w_ws)*m, n, c]  #[16, 16*16, 180]

        attn_windows = self.attn(
            x_windows, rpi=rpi_sa, mask=attn_mask
        )  # [16, 256, 180]

        if self.triangular_flag:
            attn_windows = attn_windows.view(-1, m, n, c)  # [b*(h_ws)*(w_ws), m, n, c]
            shifted_x = window_reverse_triangular(
                attn_windows, 2 * self.window_size, triangular_masks
            )  # nw*b, window_size, window_size, c #[1, 64, 64, 180]->[16, 16, 16, 180]
            shifted_x = shifted_x.view(
                b,
                h // (2 * self.window_size),
                w // (2 * self.window_size),
                c,
                2 * self.window_size,
                2 * self.window_size,
            )
            shifted_x = (
                shifted_x.permute(0, 1, 4, 2, 5, 3).contiguous().view(b, h, w, c)
            )  # [1, 64, 64, 180]
        if not self.triangular_flag:
            attn_windows = attn_windows.view(
                -1, self.window_size, self.window_size, c
            )  # [16, 16, 16, 180]
            shifted_x = window_reverse(
                attn_windows, self.window_size, h, w
            )  # b h' w' c  #[16, 16, 16, 180]->[1, 64, 64, 180]

        if self.shift_size > 0:
            attn_x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            attn_x = shifted_x

        attn_x = attn_x.view(b, h * w, c)  # [1, 64, 64, 180]->[1, 4096, 180]
        x = shortcut + self.drop_path(attn_x) + conv_x * self.conv_scale
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class HAB_S(nn.Module):
    r"""Hybrid Attention Block_Sparse.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=16,
        shift_size=0,
        interval=2,
        triangular_flag=0,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.interval = interval
        self.mlp_ratio = mlp_ratio
        self.triangular_flag = triangular_flag
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        # assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention_S(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.conv_scale = conv_scale
        self.conv_block = CAB(
            num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self, x, x_size, rpi_sa, attn_mask, triangular_masks
    ):  # [1, 4096, 180], (64, 64)
        h, w = x_size
        b, l, c = x.shape

        assert l == h * w, "input feature has wrong size %d, %d, %d" % (l, h, w)
        if min(h, w) <= self.window_size:
            self.window_size = min(
                h, w
            )  # Won't partition, if window size is larger than input resolution

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)  # [1, 64, 64, 180]
        conv_x = self.conv_block(x.permute(0, 3, 1, 2))  # [1, 180, 64, 64]
        conv_x = (
            conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        )  # [1, 4096, 180]
        size_par = self.interval
        pad_l = pad_t = 0
        pad_r = (size_par - w % size_par) % size_par
        pad_b = (size_par - h % size_par) % size_par
        x = F.pad(
            x, (0, 0, pad_l, pad_r, pad_t, pad_b)
        )  # [1, 64, 64, 180]-->[1, pad_t+64+pad_b, pad_l+64+pad_r, 0+180+0]
        _, Hd, Wd, _ = x.shape
        mask = torch.zeros((1, Hd, Wd, 1), device=x.device)
        if pad_b > 0:
            mask[:, -pad_b:, :, :] = -1
        if pad_r > 0:
            mask[:, :, -pad_r:, :] = -1

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            if self.shift_size == 8:
                attn_mask = attn_mask[0]
            if self.shift_size == 16:
                attn_mask = attn_mask[1]
            if self.shift_size == 24:
                attn_mask = attn_mask[2]
        else:
            shifted_x = x  # [1, 64, 64, 180]
            attn_mask = None

        I, Gh, Gw = self.interval, Hd // self.interval, Wd // self.interval
        shifted_sparse_x = (
            shifted_x.reshape(b, Gh, I, Gw, I, c).permute(0, 2, 4, 1, 3, 5).contiguous()
        )
        shifted_sparse_x = shifted_sparse_x.reshape(b * I * I, Gh, Gw, c)
        nP = I**2  # number of partitioning groups
        # attn_mask_sp
        if pad_r > 0 or pad_b > 0:
            mask = (
                mask.reshape(1, Gh, I, Gw, I, 1).permute(0, 2, 4, 1, 3, 5).contiguous()
            )
            mask = mask.reshape(nP, 1, Gh * Gw)
            attn_mask_sp = torch.zeros((nP, Gh * Gw, Gh * Gw), device=x.device)
            attn_mask_sp = attn_mask_sp.masked_fill(mask < 0, float("-inf"))
        else:
            attn_mask_sp = None

        if not self.triangular_flag:
            x_windows = window_partition(
                shifted_sparse_x, self.window_size
            )  # nw*b, window_size, window_size, c #[1, 64, 64, 180]->[16, 16, 16, 180]
            x_windows = x_windows.view(
                -1, self.window_size * self.window_size, c
            )  # nw*b, window_size*window_size, c  #[16, 16*16, 180]
        if self.triangular_flag:
            assert Gh >= (2 * self.window_size) and Gw >= (
                2 * self.window_size
            ), "input feature has wrong size"
            x_windows = window_partition_triangular(
                shifted_sparse_x, 2 * self.window_size, triangular_masks
            )  # nw*b, window_size, window_size, c #[1, 64, 64, 180]->[16, 16, 16, 180]
            _, m, n, _ = x_windows.shape  # [b*(h_ws)*(w_ws), m, n, c]
            x_windows = x_windows.view(
                -1, n, c
            )  # [b*(h_ws)*(w_ws)*m, n, c]  #[16, 16*16, 180]

        attn_windows = self.attn(
            x_windows, rpi=rpi_sa, mask=attn_mask, sp_mask=attn_mask_sp
        )  # [16, 256, 180]

        if self.triangular_flag:
            attn_windows = attn_windows.view(-1, m, n, c)  # [b*(h_ws)*(w_ws), m, n, c]
            shifted_sparse_x = window_reverse_triangular(
                attn_windows, 2 * self.window_size, triangular_masks
            )  # nw*b, window_size, window_size, c #[1, 64, 64, 180]->[16, 16, 16, 180]
            shifted_sparse_x = shifted_sparse_x.view(
                -1,
                Gh // (2 * self.window_size),
                Gw // (2 * self.window_size),
                c,
                2 * self.window_size,
                2 * self.window_size,
            )
            shifted_sparse_x = (
                shifted_sparse_x.permute(0, 1, 4, 2, 5, 3)
                .contiguous()
                .view(-1, Gh, Gw, c)
            )  # [1, 64, 64, 180]
        if not self.triangular_flag:
            attn_windows = attn_windows.view(
                -1, self.window_size, self.window_size, c
            )  # [16, 16, 16, 180]
            shifted_sparse_x = window_reverse(
                attn_windows, self.window_size, Gh, Gw
            )  # b h' w' c  #[16, 16, 16, 180]->[1, 64, 64, 180]

        shifted_sparse_x = (
            shifted_sparse_x.reshape(b, I, I, Gh, Gw, c)
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
        )  # b, Gh, I, Gw, I, c
        shifted_x = shifted_sparse_x.reshape(b, Hd, Wd, c)

        if self.shift_size > 0:
            attn_x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            attn_x = shifted_x

        if pad_r > 0 or pad_b > 0:
            attn_x = attn_x[:, :h, :w, :].contiguous()
        attn_x = attn_x.view(b, h * w, c)

        x = shortcut + self.drop_path(attn_x) + conv_x * self.conv_scale
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, "input feature has wrong size"
        assert h % 2 == 0 and w % 2 == 0, f"x size ({h}*{w}) are not even."

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x


class OCAB(nn.Module):
    # overlapping cross-attention block
    def __init__(
        self,
        dim,
        input_resolution,
        window_size,
        overlap_ratio,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        mlp_ratio=2,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.unfold = nn.Unfold(
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=window_size,
            padding=(self.overlap_win_size - window_size) // 2,
        )

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size + self.overlap_win_size - 1)
                * (window_size + self.overlap_win_size - 1),
                num_heads,
            )
        )
        # 2*Wh-1 * 2*Ww-1, nH
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim, dim)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU
        )

    def forward(self, x, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape
        shortcut = x

        x = self.norm1(x)
        x = x.view(b, h, w, c)

        qkv = self.qkv(x).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2)  # 3, b, c, h, w
        q = qkv[0].permute(0, 2, 3, 1)  # b, h, w, c
        kv = torch.cat((qkv[1], qkv[2]), dim=1)  # b, 2*c, h, w

        q_windows = window_partition(
            q, self.window_size
        )  # nw*b, window_size, window_size, c
        q_windows = q_windows.view(
            -1, self.window_size * self.window_size, c
        )  # nw*b, window_size*window_size, c

        kv_windows = self.unfold(kv)  # b, c*w*w, nw
        kv_windows = rearrange(
            kv_windows,
            "b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch",
            nc=2,
            ch=c,
            owh=self.overlap_win_size,
            oww=self.overlap_win_size,
        ).contiguous()
        # 2, nw*b, ow*ow, c
        k_windows, v_windows = kv_windows[0], kv_windows[1]  # nw*b, ow*ow, c

        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(
            0, 2, 1, 3
        )  # nw*b, nH, nq, d
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(
            0, 2, 1, 3
        )  # nw*b, nH, n, d
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(
            0, 2, 1, 3
        )  # nw*b, nH, n, d

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size,
            self.overlap_win_size * self.overlap_win_size,
            -1,
        )
        # ws*ws, wse*wse, nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, ws*ws, wse*wse
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, self.dim
        )
        x = window_reverse(attn_windows, self.window_size, h, w)  # b h w c
        x = x.view(b, h * w, self.dim)
        x = self.proj(x) + shortcut
        x = x + self.mlp(self.norm2(x))

        return x


class AttenBlocks(nn.Module):
    """A series of attention blocks for one RHAG.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        sparse_flag,
        depth,
        num_heads,
        window_size,
        shift_size,
        interval,
        compress_ratio,
        squeeze_factor,
        conv_scale,
        overlap_ratio,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.sparse_flag = sparse_flag
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # HAB
        if not sparse_flag:
            self.blocks = nn.ModuleList([
                HAB_D(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size[i],
                    interval=interval,
                    triangular_flag=0 if (i % 2 == 0) else 1,
                    compress_ratio=compress_ratio,
                    squeeze_factor=squeeze_factor,
                    conv_scale=conv_scale,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                HAB_S(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size[i],
                    interval=interval,
                    triangular_flag=0 if (i % 2 == 0) else 1,
                    compress_ratio=compress_ratio,
                    squeeze_factor=squeeze_factor,
                    conv_scale=conv_scale,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ])

        # OCAB
        self.overlap_attn = OCAB(
            dim=dim,
            input_resolution=input_resolution,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x, x_size, params):  # [1, 4096, 180], (64, 64)
        for blk in self.blocks:
            x = blk(
                x,
                x_size,
                params["rpi_sa"],
                params["attn_mask"],
                params["triangular_masks"],
            )

        x = self.overlap_attn(
            x, x_size, params["rpi_oca"]
        )  # [1, 4096, 180]->[1, 4096, 180]

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class RHAG(nn.Module):
    """Residual Hybrid Attention Group (RHAG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        sparse_flag,
        depth,
        num_heads,
        window_size,
        shift_size,  # tuple
        interval,  # tuple added
        compress_ratio,
        squeeze_factor,
        conv_scale,
        overlap_ratio,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        img_size=224,
        patch_size=4,
        resi_connection="1conv",
    ):
        super(RHAG, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.sparse_flag = sparse_flag

        self.residual_group = AttenBlocks(
            dim=dim,
            input_resolution=input_resolution,
            sparse_flag=sparse_flag,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            interval=interval,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "identity":
            self.conv = nn.Identity()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

    def forward(self, x, x_size, params):  # x=[2, 4096, 180], x_size=(64, 64)
        return (
            self.patch_embed(
                self.conv(
                    self.patch_unembed(self.residual_group(x, x_size, params), x_size)
                )
            )
            + x
        )


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding

    Args:
        img_size   (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans   (int): Number of input image channels. Default: 3.
        embed_dim  (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):  # [1, 180, 64, 64]
        x = x.flatten(2).transpose(
            1, 2
        )  # b Ph*Pw c  #[1, 180, 64, 64]->[1, 180, 4096]->[1, 4096, 180]
        if self.norm is not None:
            x = self.norm(x)  # [1, 4096, 180]
        return x


class PatchUnEmbed(nn.Module):
    r"""Image to Patch Unembedding

    Args:
        img_size   (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans   (int): Number of input image channels. Default: 3.
        embed_dim  (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        )  # b Ph*Pw c
        return x


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log2(scale))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(
                f"scale {scale} is not supported. " "Supported scales: 2^n and 3."
            )
        super(Upsample, self).__init__(*m)


@ARCH_REGISTRY.register()
class cfat(nn.Module):
    r"""Composite Fusion Attention Transformer
    Args:
        img_size       (int | tuple(int)): Input image size. Default 64
        patch_size     (int | tuple(int)): Patch size. Default: 1
        in_chans       (int): Number of input image channels. Default: 3
        embed_dim      (int): Patch embedding dimension. Default: 96
        depths         (tuple(int)): Depth of each Swin Transformer layer.
        num_heads      (tuple(int)): Number of attention heads in different layers.
        window_size    (int): Window size. Default: 7
        sparse_flag    (bool): Whether execute dense attention or sparse attention
        interval       (tuple(int)): Dilation in Sparse Attention
        mlp_ratio      (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias       (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale       (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate      (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer     (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape            (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm     (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale:       (int): Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(
        self,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=180,
        depths=(8, 8, 8, 8, 8),
        num_heads=(6, 6, 6, 6, 6),
        window_size=16,
        shift_size=(0, 0, 8, 8, 16, 16, 24, 24),  # changed to tuple
        interval=(0, 2, 0, 2, 0),
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        mlp_ratio=2,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=upscale,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        **kwargs,
    ):
        super(cfat, self).__init__()

        self.window_size = window_size
        self.shift_size = shift_size  # changed to tuple
        self.overlap_ratio = overlap_ratio
        self.upscale = upscale
        self.upsampler = upsampler
        self.num_layers = len(depths)  # 5
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64

        # ----Mean_Operation----######
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        # relative position index
        # ----Relative_Position_Index----######
        relative_position_index_SA = self.calculate_rpi_sa()
        relative_position_index_OCA = self.calculate_rpi_oca()
        self.register_buffer("relative_position_index_SA", relative_position_index_SA)
        self.register_buffer("relative_position_index_OCA", relative_position_index_OCA)

        # ----Shallow_Feature_Extraction----######
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ----Patch_Embedding----######
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # ----Patch_Unembedding----######
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # ----Absolute_Position_Embedding----######
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )  # (1, 4096, 180)
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # ----DropOut_with_stochastic----######
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ----Residual_Hybrid_Attention_Groups(RHAG)----######
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHAG(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                sparse_flag=0 if (i_layer % 2 == 0) else 1,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                shift_size=shift_size,  # tuple added
                interval=interval[i_layer],  # tuple added
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                overlap_ratio=overlap_ratio,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        # ----Last_Convolution+Reconstruction----######
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "identity":
            self.conv_after_body = nn.Identity()

        if self.upsampler == "pixelshuffle":
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    # ----Weight_Initialization----######
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ----Relative_Position_Index_for_HAB----######
    def calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    # ----Relative_Position_Index_for_OCAB----######
    def calculate_rpi_oca(self):
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)
        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # 2, ws, ws
        coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws
        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # 2, wse, wse
        coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse

        relative_coords = (
            coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]
        )  # 2, ws*ws, wse*wse
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # ws*ws, wse*wse, 2
        relative_coords[:, :, 0] += (
            window_size_ori - window_size_ext + 1
        )  # shift to start from 0
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1
        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    # ----Attention_Mask_for_HAB(SW-MSA)----######
    def calculate_mask(self, x_size, shift_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -shift_size),
            slice(-shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -shift_size),
            slice(-shift_size, None),
        )
        cnt = 0
        for h_s in h_slices:
            for w_s in w_slices:
                img_mask[:, h_s, w_s, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, 0.0
        )
        return attn_mask

    # ------Triangular_Window_Mask------###############
    def triangle_masks(self, x):
        ws = 2 * self.window_size
        rows = torch.arange(ws).unsqueeze(1).repeat(1, ws)
        cols = torch.arange(ws).unsqueeze(0).repeat(ws, 1)

        upper_triangle_mask = (cols > rows) & (rows + cols < ws)
        right_triangle_mask = (cols >= rows) & (rows + cols >= ws)
        bottom_triangle_mask = (cols < rows) & (rows + cols >= ws - 1)
        left_triangle_mask = (cols <= rows) & (rows + cols < ws - 1)

        return [
            upper_triangle_mask.to(x.device),
            right_triangle_mask.to(x.device),
            bottom_triangle_mask.to(x.device),
            left_triangle_mask.to(x.device),
        ]

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):  # [1, 180, 64, 64]
        x_size = (x.shape[2], x.shape[3])

        # Calculate attention mask and relative position index in advance to speed up inference.
        # The original code is very time-cosuming for large window size.
        attn_mask = tuple([
            self.calculate_mask(x_size, shift_size).to(x.device)
            for shift_size in (8, 16, 24)
        ])  # [16, 256, 256]   #changed to tuple
        triangular_masks = tuple(
            self.triangle_masks(x)
        )  # [16, 256, 256]   #changed to tuple

        params = {
            "attn_mask": attn_mask,
            "triangular_masks": triangular_masks,
            "rpi_sa": self.relative_position_index_SA,
            "rpi_oca": self.relative_position_index_OCA,
        }

        # Embed$$Unembed
        x = self.patch_embed(x)  # [1, 180, 64, 64]->[1, 4096, 180]
        if self.ape:
            x = x + self.absolute_pos_embed  # [1, 4096, 180]
        x = self.pos_drop(x)  # [1, 4096, 180]
        for layer in self.layers:
            x = layer(x, x_size, params)  # [1, 4096, 180]
        x = self.norm(x)  # b seq_len c     #[1, 4096, 180]
        x = self.patch_unembed(x, x_size)  # [1, 4096, 180]->[1, 180, 64, 64]

        return x

    def forward(self, x):  # [1, 3, 64, 64]
        self.mean = self.mean.type_as(x)  # [1, 3, 1, 1]
        x = (x - self.mean) * self.img_range  # [1, 3, 64, 64]

        if self.upsampler == "pixelshuffle":
            # for classical SR
            x = self.conv_first(x)  # [1, 180, 64, 64]
            x = self.conv_after_body(self.forward_features(x)) + x  # [1, 180, 64, 64]
            x = self.conv_before_upsample(x)  # [1, 64, 64, 64]
            x = self.conv_last(self.upsample(x))  # [1, 3, 256, 256]

        x = x / self.img_range + self.mean  # [1, 3, 256, 256]

        return x
