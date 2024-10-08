# type: ignore  # noqa: PGH003
import math

import numpy as np
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_

from neosr.archs.arch_util import DropPath, net_opt
from neosr.utils.registry import ARCH_REGISTRY

upscale, __ = net_opt()


def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    return (
        img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    )


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    return img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class MatMul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a @ b


class SpatialGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x, H, W):
        # Split
        x1, x2 = x.chunk(2, dim=-1)
        B, _N, C = x.shape
        x2 = (
            self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W))
            .flatten(2)
            .transpose(-1, -2)
            .contiguous()
        )

        return x1 * x2


class MLP(nn.Module):
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
        self.sg = SpatialGate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        return self.drop(x)


class DynamicPosBias(nn.Module):
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
            pos = self.pos_proj(biases)
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class Window_Attention(nn.Module):
    """Spatial Window Self-Attention.
    It supports rectangle window (containing square window).
    Args:
        dim (int): Number of input channels.
        idx (int): The indentix of window. (0/1)
        split_size (tuple(int)): Height and Width of spatial window.
        dim_out (int | None): The dimension of the attention output. Default: None
        num_heads (int): Number of attention heads. Default: 6
        attn_drop (float): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float): Dropout ratio of output. Default: 0.0
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set
        position_bias (bool): The dynamic relative position bias. Default: True
    """

    def __init__(
        self,
        dim,
        idx,
        split_size=None,
        dim_out=None,
        num_heads=6,
        attn_drop=0.0,
        proj_drop=0.0,
        qk_scale=None,
        position_bias=True,
    ):
        if split_size is None:
            split_size = [8, 8]
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)

            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w], indexing="ij"))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer("rpe_biases", biases)

            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, _N, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        return (
            x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def forward(self, qkv, H, W, mask=None):
        """
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # B head N C @ B head C N --> B head N N

        # calculate drpe
        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1
            )
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0
            )
            attn = attn.view(-1, self.num_heads, N, N)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(
            -1, self.H_sp * self.W_sp, C
        )  # B head N N @ B head N C

        # merge the window, window to image
        return windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C


class Adaptive_Window_Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        reso=64,
        split_size=None,
        shift_size=None,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        rg_idx=0,
        b_idx=0,
    ):
        if shift_size is None:
            shift_size = [1, 2]
        if split_size is None:
            split_size = [8, 8]
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.b_idx = b_idx
        self.rg_idx = rg_idx
        self.patches_resolution = reso
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        assert (
            0 <= self.shift_size[0] < self.split_size[0]
        ), "shift_size must in 0-split_size0"
        assert (
            0 <= self.shift_size[1] < self.split_size[1]
        ), "shift_size must in 0-split_size1"

        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
            Window_Attention(
                dim // 2,
                idx=i,
                split_size=split_size,
                num_heads=num_heads // 2,
                dim_out=dim // 2,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                position_bias=True,
            )
            for i in range(self.branch_num)
        ])

        if (self.rg_idx % 2 == 0 and self.b_idx > 0 and (self.b_idx - 2) % 4 == 0) or (
            self.rg_idx % 2 != 0 and self.b_idx % 4 == 0
        ):
            attn_mask = self.calculate_mask(
                self.patches_resolution, self.patches_resolution
            )
            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            attn_mask = None
            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

    def calculate_mask(self, H, W):
        # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # calculate attention mask for shift window
        img_mask_0 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=0
        img_mask_1 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=1
        h_slices_0 = (
            slice(0, -self.split_size[0]),
            slice(-self.split_size[0], -self.shift_size[0]),
            slice(-self.shift_size[0], None),
        )
        w_slices_0 = (
            slice(0, -self.split_size[1]),
            slice(-self.split_size[1], -self.shift_size[1]),
            slice(-self.shift_size[1], None),
        )

        h_slices_1 = (
            slice(0, -self.split_size[1]),
            slice(-self.split_size[1], -self.shift_size[1]),
            slice(-self.shift_size[1], None),
        )
        w_slices_1 = (
            slice(0, -self.split_size[0]),
            slice(-self.split_size[0], -self.shift_size[0]),
            slice(-self.shift_size[0], None),
        )
        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt += 1
        cnt = 0
        for h in h_slices_1:
            for w in w_slices_1:
                img_mask_1[:, h, w, :] = cnt
                cnt += 1

        # calculate mask for window-0
        img_mask_0 = img_mask_0.view(
            1,
            H // self.split_size[0],
            self.split_size[0],
            W // self.split_size[1],
            self.split_size[1],
            1,
        )
        img_mask_0 = (
            img_mask_0.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, self.split_size[0], self.split_size[1], 1)
        )  # nW, sw[0], sw[1], 1
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, -100.0).masked_fill(
            attn_mask_0 == 0, 0.0
        )

        # calculate mask for window-1
        img_mask_1 = img_mask_1.view(
            1,
            H // self.split_size[1],
            self.split_size[1],
            W // self.split_size[0],
            self.split_size[0],
            1,
        )
        img_mask_1 = (
            img_mask_1.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, self.split_size[1], self.split_size[0], 1)
        )  # nW, sw[1], sw[0], 1
        mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, -100.0).masked_fill(
            attn_mask_1 == 0, 0.0
        )

        return attn_mask_0, attn_mask_1

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)  # 3, B, HW, C
        qkv[2].transpose(-2, -1).contiguous().view(B, C, H, W)

        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_l = pad_t = 0
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size

        qkv = qkv.reshape(3 * B, H, W, C).permute(0, 3, 1, 2)  # 3B C H W
        qkv = (
            F.pad(qkv, (pad_l, pad_r, pad_t, pad_b))
            .reshape(3, B, C, -1)
            .transpose(-2, -1)
        )  # l r t b
        _H = pad_b + H
        _W = pad_r + W
        _L = _H * _W

        if (self.rg_idx % 2 == 0 and self.b_idx > 0 and (self.b_idx - 2) % 4 == 0) or (
            self.rg_idx % 2 != 0 and self.b_idx % 4 == 0
        ):
            qkv = qkv.view(3, B, _H, _W, C)
            qkv_0 = torch.roll(
                qkv[:, :, :, :, : C // 2],
                shifts=(-self.shift_size[0], -self.shift_size[1]),
                dims=(2, 3),
            )
            qkv_0 = qkv_0.view(3, B, _L, C // 2)
            qkv_1 = torch.roll(
                qkv[:, :, :, :, C // 2 :],
                shifts=(-self.shift_size[1], -self.shift_size[0]),
                dims=(2, 3),
            )
            qkv_1 = qkv_1.view(3, B, _L, C // 2)

            if self.patches_resolution != _H or self.patches_resolution != _W:
                mask_tmp = self.calculate_mask(_H, _W)
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=mask_tmp[0].to(x.device))
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=mask_tmp[1].to(x.device))
            else:
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=self.attn_mask_0)
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=self.attn_mask_1)

            x1 = torch.roll(
                x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2)
            )
            x2 = torch.roll(
                x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2)
            )
            x1 = x1[:, :H, :W, :].reshape(B, L, C // 2)
            x2 = x2[:, :H, :W, :].reshape(B, L, C // 2)

            attened_x = torch.cat([x1, x2], dim=2)

        else:
            x1 = self.attns[0](qkv[:, :, :, : C // 2], _H, _W)[:, :H, :W, :].reshape(
                B, L, C // 2
            )
            x2 = self.attns[1](qkv[:, :, :, C // 2 :], _H, _W)[:, :H, :W, :].reshape(
                B, L, C // 2
            )
            # attention output
            attened_x = torch.cat([x1, x2], dim=2)

        x = attened_x

        x = self.proj(x)
        return self.proj_drop(x)


class Linear_Attention(nn.Module):
    def __init__(
        self,
        in_channels,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        res_kernel_size=9,
        sparse_reg=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = in_channels // num_heads
        self.scale = head_dim**-0.5
        self.sparse_reg = sparse_reg

        self.qkv = nn.Linear(in_channels, in_channels * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_channels, in_channels)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kq_matmul = MatMul()
        self.kqv_matmul = MatMul()
        if self.sparse_reg:
            self.qk_matmul = MatMul()
            self.sv_matmul = MatMul()

        self.dconv = nn.Conv2d(
            in_channels=self.num_heads,
            out_channels=self.num_heads,
            kernel_size=(res_kernel_size, 1),
            padding=(res_kernel_size // 2, 0),
            bias=False,
            groups=self.num_heads,
        )

    def forward(self, x):
        N, L, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(N, L, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        if self.sparse_reg:
            attn = self.qk_matmul(q * self.scale, k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            mask = attn > 0.02
            sparse = mask * attn

        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        dconv_v = self.dconv(v)

        attn = self.kq_matmul(k.transpose(-2, -1), v)

        if self.sparse_reg:
            x = (
                self.sv_matmul(sparse, v)
                + 0.5 * v
                + 1.0 / math.pi * self.kqv_matmul(q, attn)
            )
        else:
            x = 0.5 * v + 1.0 / math.pi * self.kqv_matmul(q, attn)
        x = x / x.norm(dim=-1, keepdim=True)
        x += dconv_v
        x = x.transpose(1, 2).reshape(N, L, C)
        x = self.proj(x)
        return self.proj_drop(x)


class DCTB(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        reso=64,
        split_size=None,
        shift_size=None,
        expansion_factor=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        rg_idx=0,
        b_idx=0,
    ):
        if shift_size is None:
            shift_size = [1, 2]
        if split_size is None:
            split_size = [2, 4]
        super().__init__()

        self.norm1 = norm_layer(dim // 2)

        if b_idx % 2 == 0:
            self.swsa = Adaptive_Window_Attention(
                dim // 2,
                num_heads=num_heads // 2,
                reso=reso,
                split_size=split_size,
                shift_size=shift_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                rg_idx=rg_idx,
                b_idx=b_idx,
            )

            self.lla = Linear_Attention(in_channels=dim // 2, num_heads=num_heads // 2)
        else:
            self.fc1 = nn.Conv2d(dim, dim // 4, kernel_size=1)
            self.dwc = nn.Conv2d(dim // 4, dim // 4, kernel_size=3, stride=1, padding=1)
            self.fc2 = nn.Conv2d(dim // 4, dim, kernel_size=1)

            self.eca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim // 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 16, dim, 1),
                nn.Sigmoid(),
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        ffn_hidden_dim = int(dim * expansion_factor)
        self.ffn = MLP(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
        )
        self.norm2 = norm_layer(dim)

    def forward(self, x, x_size, prev_attention=None):  #
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        H, W = x_size
        _, _, C = x.shape
        if prev_attention is None:
            swsa_x = self.swsa(self.norm1(x[:, :, : C // 2]), H, W)
            lla_x = self.norm1(x[:, :, C // 2 :])
            lla_x = self.lla(lla_x)
            out_x = torch.cat([swsa_x, lla_x], dim=-1)
        else:
            s_x = self.norm1(x[:, :, : C // 2])
            l_x = self.norm1(x[:, :, C // 2 :])
            x = torch.cat([s_x, l_x], dim=-1)

            sa_output = prev_attention

            # FC1
            out = rearrange(sa_output, "b (h w) c -> b c h w", h=H, w=W).contiguous()

            out = self.fc1(out)
            out = F.gelu(self.dwc(out))
            out = self.fc2(out)
            _map = self.eca(out)
            out = out * _map
            out_x = rearrange(out, "b c h w -> b (h w) c").contiguous()  # [1,4096,180]
        # DwC
        x = x + self.drop_path(out_x)
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x, out_x


class ResidualGroup(nn.Module):
    """ResidualGroup
    Args:
        dim (int): Number of input channels.
        reso (int): Input resolution.
        num_heads (int): Number of attention heads.
        split_size (tuple(int)): Height and Width of spatial window.
        expansion_factor (float): Ratio of ffn hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop (float): Dropout rate. Default: 0
        attn_drop(float): Attention dropout rate. Default: 0
        drop_paths (float | None): Stochastic depth rate.
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        depth (int): Number of dual aggregation Transformer blocks in residual group.
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(
        self,
        dim,
        reso,
        num_heads,
        split_size=None,
        expansion_factor=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_paths=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        depth=2,
        resi_connection="1conv",
        rg_idx=0,
    ):
        if split_size is None:
            split_size = [2, 4]
        super().__init__()
        self.reso = reso

        self.blocks = nn.ModuleList([
            DCTB(
                dim=dim,
                num_heads=num_heads,
                reso=reso,
                split_size=split_size,
                shift_size=[split_size[0] // 2, split_size[1] // 2],
                expansion_factor=expansion_factor,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_paths[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                rg_idx=rg_idx,
                b_idx=i,
            )
            for i in range(depth)
        ])

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

    def forward(self, x, x_size):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        H, W = x_size
        res = x
        r_idx = 0

        prev_attention = None  #
        for blk in self.blocks:
            # print('**0**')
            if r_idx % 2 == 0:
                x, prev_attention = blk(x, x_size, None)
            else:
                x, _ = blk(x, x_size, prev_attention)
                prev_attention = None
            r_idx = r_idx + 1

        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.conv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return res + x


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
                m.extend((
                    nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
                    nn.PixelShuffle(2),
                ))
        elif scale == 3:
            m.extend((nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1), nn.PixelShuffle(3)))
        else:
            msg = f"scale {scale} is not supported. " "Supported scales: 2^n and 3."
            raise ValueError(msg)
        super().__init__(*m)


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
        m.extend((
            nn.Conv2d(num_feat, scale**2 * num_out_ch, 3, 1, 1),
            nn.PixelShuffle(scale),
        ))
        super().__init__(*m)


@ARCH_REGISTRY.register()
class dct(nn.Module):
    """Dual Compression Transformer
    Args:
        img_size (int): Input image size. Default: 64
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 180
        depths (tuple(int)): Depth of each residual group (number of DCTB in each RG).
        split_size (tuple(int)): Height and Width of spatial window.
        num_heads (tuple(int)): Number of attention heads in different residual groups.
        expansion_factor (float): Ratio of ffn hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        upscale: Upscale factor. 2/3/4 for image SR
        img_range: Image range. 1. or 255.
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(
        self,
        img_size=64,
        in_chans=3,
        embed_dim=80,
        split_size=None,
        depth=[20],
        num_heads=[8],
        expansion_factor=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        upscale=upscale,
        img_range=1.0,
        resi_connection="3conv",
        upsampler="pixelshuffledirect",
        **kwargs,
    ):
        if num_heads is None:
            num_heads = [2, 2, 2, 2]
        if depth is None:
            depth = [2, 2, 2, 2]
        if split_size is None:
            split_size = [2, 4]
        super().__init__()

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

        # ------------------------- 1, Shallow Feature Extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, Deep Feature Extraction ------------------------- #
        self.num_layers = len(depth)
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        heads = num_heads

        self.before_RG = nn.Sequential(
            Rearrange("b c h w -> b (h w) c"), nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))
        ]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = ResidualGroup(
                dim=embed_dim,
                num_heads=heads[i],
                reso=img_size,
                split_size=split_size,
                expansion_factor=expansion_factor,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_paths=dpr[sum(depth[:i]) : sum(depth[: i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                depth=depth[i],
                resi_connection=resi_connection,
                rg_idx=i,
            )
            self.layers.append(layer)

        self.norm = norm_layer(curr_dim)
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

        # ------------------------- 3, Reconstruction ------------------------- #
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
                upscale, embed_dim, num_out_ch, (img_size, img_size)
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(
            m, nn.LayerNorm | nn.BatchNorm2d | nn.GroupNorm | nn.InstanceNorm2d
        ):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        _, _, H, W = x.shape
        x_size = [H, W]
        x = self.before_RG(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        return rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == "pixelshuffle":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffledirect":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)

        return x / self.img_range + self.mean
