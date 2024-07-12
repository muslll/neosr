import math

import torch
import torch.nn.functional as F
from torch.utils import checkpoint
from torch import nn
from torch.nn.init import trunc_normal_

from neosr.archs.arch_util import DropPath, net_opt, to_2tuple
from neosr.utils.registry import ARCH_REGISTRY

upscale, __ = net_opt()


class DFE(nn.Module):
    """Dual Feature Extraction
    Args:
        in_features (int): Number of input channels.
        out_features (int): Number of output channels.
    """

    def __init__(self, in_features, out_features):
        super().__init__()

        self.out_features = out_features

        self.conv = nn.Sequential(
            nn.Conv2d(in_features, in_features // 5, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_features // 5, in_features // 5, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_features // 5, out_features, 1, 1, 0),
        )

        self.linear = nn.Conv2d(in_features, out_features, 1, 1, 0)

    def forward(self, x, x_size):
        B, L, C = x.shape
        H, W = x_size
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        x = self.conv(x) * self.linear(x)
        x = x.view(B, -1, H * W).permute(0, 2, 1).contiguous()

        return x


class Mlp(nn.Module):
    """MLP-based Feed-Forward Network
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """

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


class dwconv(nn.Module):
    def __init__(self, hidden_features):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                hidden_features,
                hidden_features,
                kernel_size=5,
                stride=1,
                padding=2,
                dilation=1,
                groups=hidden_features,
            ),
            nn.GELU(),
        )
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = (
            x.transpose(1, 2)
            .view(x.shape[0], self.hidden_features, x_size[0], x_size[1])
            .contiguous()
        )  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
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
        self.dwconv = dwconv(hidden_features=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (tuple): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(
        B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size[0], window_size[1], C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (tuple): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] * (window_size[0] * window_size[1]) / (H * W))
    x = windows.view(
        B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of heads for spatial self-correlation.
        residual (bool):  If True, use residual strage to connect conv.
    """

    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads),
        )

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)  # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class SCC(nn.Module):
    """Spatial-Channel Correlation.
    Args:
        dim (int): Number of input channels.
        base_win_size (tuple[int]): The height and width of the base window.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of heads for spatial self-correlation.
        value_drop (float, optional): Dropout ratio of value. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self, dim, base_win_size, window_size, num_heads, value_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        # parameters
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        # feature projection
        self.qv = DFE(dim, dim)
        self.proj = nn.Linear(dim, dim)

        # dropout
        self.value_drop = nn.Dropout(value_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # base window size
        min_h = min(self.window_size[0], base_win_size[0])
        min_w = min(self.window_size[1], base_win_size[1])
        self.base_win_size = (min_h, min_w)

        # normalization factor and spatial linear layer for S-SC
        head_dim = dim // (2 * num_heads)
        self.scale = head_dim
        self.spatial_linear = nn.Linear(
            self.window_size[0]
            * self.window_size[1]
            // (self.base_win_size[0] * self.base_win_size[1]),
            1,
        )

        # define a parameter table of relative position bias
        self.H_sp, self.W_sp = self.window_size
        self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)

    def spatial_linear_projection(self, x):
        B, num_h, L, C = x.shape
        H, W = self.window_size
        map_H, map_W = self.base_win_size

        x = (
            x.view(B, num_h, map_H, H // map_H, map_W, W // map_W, C)
            .permute(0, 1, 2, 4, 6, 3, 5)
            .contiguous()
            .view(B, num_h, map_H * map_W, C, -1)
        )
        x = self.spatial_linear(x).view(B, num_h, map_H * map_W, C)
        return x

    def spatial_self_correlation(self, q, v):
        B, num_head, L, C = q.shape

        # spatial projection
        v = self.spatial_linear_projection(v)

        # compute correlation map
        corr_map = (q @ v.transpose(-2, -1)) / self.scale

        # add relative position bias
        # generate mother-set
        position_bias_h = torch.arange(1 - self.H_sp, self.H_sp, device=v.device)
        position_bias_w = torch.arange(1 - self.W_sp, self.W_sp, device=v.device)
        biases = torch.stack(
            torch.meshgrid([position_bias_h, position_bias_w], indexing="ij")
        )
        rpe_biases = biases.flatten(1).transpose(0, 1).contiguous().float()
        pos = self.pos(rpe_biases)

        # select position bias
        coords_h = torch.arange(self.H_sp, device=v.device)
        coords_w = torch.arange(self.W_sp, device=v.device)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.H_sp - 1
        relative_coords[:, :, 1] += self.W_sp - 1
        relative_coords[:, :, 0] *= 2 * self.W_sp - 1
        relative_position_index = relative_coords.sum(-1)
        relative_position_bias = pos[relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.base_win_size[0],
            self.window_size[0] // self.base_win_size[0],
            self.base_win_size[1],
            self.window_size[1] // self.base_win_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = (
            relative_position_bias.permute(0, 1, 3, 5, 2, 4)
            .contiguous()
            .view(
                self.window_size[0] * self.window_size[1],
                self.base_win_size[0] * self.base_win_size[1],
                self.num_heads,
                -1,
            )
            .mean(-1)
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        corr_map = corr_map + relative_position_bias.unsqueeze(0)

        # transformation
        v_drop = self.value_drop(v)
        x = (corr_map @ v_drop).permute(0, 2, 1, 3).contiguous().view(B, L, -1)

        return x

    def channel_self_correlation(self, q, v):
        B, num_head, L, C = q.shape

        # apply single head strategy
        q = q.permute(0, 2, 1, 3).contiguous().view(B, L, num_head * C)
        v = v.permute(0, 2, 1, 3).contiguous().view(B, L, num_head * C)

        # compute correlation map
        corr_map = (q.transpose(-2, -1) @ v) / L

        # transformation
        v_drop = self.value_drop(v)
        x = (
            (corr_map @ v_drop.transpose(-2, -1))
            .permute(0, 2, 1)
            .contiguous()
            .view(B, L, -1)
        )

        return x

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        xB, xH, xW, xC = x.shape
        qv = self.qv(x.view(xB, -1, xC), (xH, xW)).view(xB, xH, xW, xC)

        # window partition
        qv = window_partition(qv, self.window_size)
        qv = qv.view(-1, self.window_size[0] * self.window_size[1], xC)

        # qv splitting
        B, L, C = qv.shape
        qv = (
            qv.view(B, L, 2, self.num_heads, C // (2 * self.num_heads))
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        q, v = qv[0], qv[1]  # B, num_heads, L, C//num_heads

        # spatial self-correlation (S-SC)
        x_spatial = self.spatial_self_correlation(q, v)
        x_spatial = x_spatial.view(-1, self.window_size[0], self.window_size[1], C // 2)
        x_spatial = window_reverse(
            x_spatial, (self.window_size[0], self.window_size[1]), xH, xW
        )  # xB xH xW xC

        # channel self-correlation (C-SC)
        x_channel = self.channel_self_correlation(q, v)
        x_channel = x_channel.view(-1, self.window_size[0], self.window_size[1], C // 2)
        x_channel = window_reverse(
            x_channel, (self.window_size[0], self.window_size[1]), xH, xW
        )  # xB xH xW xC

        # spatial-channel information fusion
        x = torch.cat([x_spatial, x_channel], -1)
        x = self.proj_drop(self.proj(x))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"


class HierarchicalTransformerBlock(nn.Module):
    """Hierarchical Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of heads for spatial self-correlation.
        base_win_size (tuple[int]): The height and width of the base window.
        window_size (tuple[int]): The height and width of the window.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        value_drop (float, optional): Dropout ratio of value. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        base_win_size,
        window_size,
        mlp_ratio=4.0,
        drop=0.0,
        value_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        # check window size
        if (window_size[0] > base_win_size[0]) and (window_size[1] > base_win_size[1]):
            assert (
                window_size[0] % base_win_size[0] == 0
            ), "please ensure the window size is smaller than or divisible by the base window size"
            assert (
                window_size[1] % base_win_size[1] == 0
            ), "please ensure the window size is smaller than or divisible by the base window size"

        self.norm1 = norm_layer(dim)
        self.correlation = SCC(
            dim,
            base_win_size=base_win_size,
            window_size=self.window_size,
            num_heads=num_heads,
            value_drop=value_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvFFN(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def check_image_size(self, x, win_size):
        x = x.permute(0, 3, 1, 2).contiguous()
        _, _, h, w = x.size()
        mod_pad_h = (win_size[0] - h % win_size[0]) % win_size[0]
        mod_pad_w = (win_size[1] - w % win_size[1]) % win_size[1]

        if mod_pad_h >= h or mod_pad_w >= w:
            pad_h, pad_w = h - 1, w - 1
            x = F.pad(x, (0, pad_w, 0, pad_h), "reflect")
        else:
            pad_h, pad_w = 0, 0

        mod_pad_h = mod_pad_h - pad_h
        mod_pad_w = mod_pad_w - pad_w

        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def forward(self, x, x_size, win_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = x.view(B, H, W, C)

        # padding
        x = self.check_image_size(x, win_size)
        _, H_pad, W_pad, _ = x.shape  # shape after padding

        x = self.correlation(x)

        # unpad
        x = x[:, :H, :W, :].contiguous()

        # norm
        x = x.view(B, H * W, C)
        x = self.norm1(x)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.norm2(self.mlp(x, x_size)))

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"
        )


class PatchMerging(nn.Module):
    """Patch Merging Layer.
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
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer(nn.Module):
    """A basic Hierarchical Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of heads for spatial self-correlation.
        base_win_size (tuple[int]): The height and width of the base window.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        value_drop (float, optional): Dropout ratio of value. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        hier_win_ratios (list): hierarchical window ratios for a transformer block. Default: [0.5,1,2,4,6,8].
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        base_win_size,
        mlp_ratio=4.0,
        drop=0.0,
        value_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        hier_win_ratios=[0.5, 1, 2, 4, 6, 8],
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.win_hs = [int(base_win_size[0] * ratio) for ratio in hier_win_ratios]
        self.win_ws = [int(base_win_size[1] * ratio) for ratio in hier_win_ratios]

        # build blocks
        self.blocks = nn.ModuleList([
            HierarchicalTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                base_win_size=base_win_size,
                window_size=(self.win_hs[i], self.win_ws[i]),
                mlp_ratio=mlp_ratio,
                drop=drop,
                value_drop=value_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x, x_size):
        i = 0
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(
                    blk, x, x_size, (self.win_hs[i], self.win_ws[i])
                )
            else:
                x = blk(x, x_size, (self.win_hs[i], self.win_ws[i]))
            i = i + 1

        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class RHTB(nn.Module):
    """Residual Hierarchical Transformer Block (RHTB).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of heads for spatial self-correlation.
        base_win_size (tuple[int]): The height and width of the base window.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        value_drop (float, optional): Dropout ratio of value. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
        hier_win_ratios (list): hierarchical window ratios for a transformer block. Default: [0.5,1,2,4,6,8].
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        base_win_size,
        mlp_ratio=4.0,
        drop=0.0,
        value_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        img_size=224,
        patch_size=4,
        resi_connection="1conv",
        hier_win_ratios=[0.5, 1, 2, 4, 6, 8],
    ):
        super(RHTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            base_win_size=base_win_size,
            mlp_ratio=mlp_ratio,
            drop=drop,
            value_drop=value_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            hier_win_ratios=hier_win_ratios,
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

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

    def forward(self, x, x_size):
        return (
            self.patch_embed(
                self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))
            )
            + x
        )


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
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

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    r"""Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
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
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
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


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
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


@ARCH_REGISTRY.register()
class hit_srf(nn.Module):
    """HiT-SRF network.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Transformer block.
        num_heads (tuple(int)): Number of heads for spatial self-correlation in different layers.
        base_win_size (tuple[int]): The height and width of the base window.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_rate (float): Dropout rate. Default: 0
        value_drop_rate (float): Dropout ratio of value. Default: 0.0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale (int): Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range (float): Image range. 1. or 255.
        upsampler (str): The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection (str): The convolutional block before residual connection. '1conv'/'3conv'
        hier_win_ratios (list): hierarchical window ratios for a transformer block. Default: [0.5,1,2,4,6,8].
    """

    def __init__(
        self,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=60,
        depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6),
        base_win_size=[8, 8],
        mlp_ratio=2.0,
        drop_rate=0.0,
        value_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=upscale,
        img_range=1.0,
        upsampler="pixelshuffledirect",
        resi_connection="1conv",
        hier_win_ratios=[0.5, 1, 2, 4, 6, 8],
        **kwargs,
    ):
        super(hit_srf, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.5, 0.5, 0.5)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.base_win_size = base_win_size

        #####################################################################################################
        # 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        #####################################################################################################
        # 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
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

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build Residual Hierarchical Transformer blocks (RHTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                base_win_size=base_win_size,
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                value_drop=value_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                hier_win_ratios=hier_win_ratios,
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )

        #####################################################################################################
        # 3, high quality image reconstruction ################################
        if self.upsampler == "pixelshuffle":
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(
                upscale,
                embed_dim,
                num_out_ch,
                (patches_resolution[0], patches_resolution[1]),
            )
        elif self.upsampler == "nearest+conv":
            # for real-world SR (less artifacts)
            assert self.upscale == 4, "only support x4 now."
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == "pixelshuffle":
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == "nearest+conv":
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(
                self.conv_up1(
                    torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
                )
            )
            x = self.lrelu(
                self.conv_up2(
                    torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
                )
            )
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, : H * self.upscale, : W * self.upscale]


@ARCH_REGISTRY.register()
def hit_srf_medium(**kwargs):
    return hit_srf(
        img_size=48,
        base_win_size=[16, 16],
        depths=(6, 6, 6, 6, 6, 6),
        embed_dim=180,
        num_heads=(6, 6, 6, 6, 6, 6),
        upsampler="pixelshuffle",
        **kwargs,
    )


@ARCH_REGISTRY.register()
def hit_srf_large(**kwargs):
    return hit_srf(
        img_size=48,
        base_win_size=[24, 24],
        depths=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        embed_dim=180,
        num_heads=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        upsampler="pixelshuffle",
        **kwargs,
    )
