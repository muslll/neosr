# type: ignore  # noqa: PGH003
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from neosr.archs.arch_util import DropPath, net_opt
from neosr.utils.registry import ARCH_REGISTRY

upscale, __ = net_opt()


class dwconv(nn.Module):
    # dwconv is used by FFN to interact with spatial information
    def __init__(self, hidden_features):
        super().__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                hidden_features,
                hidden_features,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=hidden_features,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_features,
                hidden_features,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                groups=hidden_features,
            ),
        )
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = (
            x.transpose(1, 2)
            .view(x.shape[0], self.hidden_features, x_size[0], x_size[1])
            .contiguous()
        )  # b Ph*Pw c
        x = self.depthwise_conv(x)
        return x.flatten(2).transpose(1, 2).contiguous()


class FFN(nn.Module):
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
        self.act = act_layer()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = dwconv(hidden_features=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, x_size):
        b_, _, _c = x.shape
        _h, _w = x_size
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(
        b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c
    )
    return (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size[0], window_size[1], c)
    )


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size[0] / window_size[1]))
    x = windows.view(
        b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class GRSA(nn.Module):
    """
    The core of our GRFormer. GRSA incorporates two components: GRL and ESRPB. The former reduces the amount of
    parameters and calculation with the performance remaining unchanged, the latter better represent the position information,
    thus improving the performance of the model.
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
        self.dim = dim
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.num_heads = num_heads
        dim // num_heads
        # self.scale = qk_scale or head_dim**-0.5

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )
        # mlp to generate continuous relative position bias
        self.ESRPB_MLP = nn.Sequential(
            nn.Linear(2, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_heads, bias=False),
        )
        # get relative_coords_table
        relative_coords_h = torch.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32
        )
        relative_coords_w = torch.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32
        )
        relative_position_bias_table = (
            torch.stack(
                torch.meshgrid([relative_coords_h, relative_coords_w], indexing="ij")
            )
            .permute(1, 2, 0)
            .contiguous()
            .unsqueeze(0)
        )  # 1, 2*Wh-1, 2*Ww-1, 2
        relative_position_bias_table[:, :, :, 0] /= self.window_size[0] - 1
        relative_position_bias_table[:, :, :, 1] /= self.window_size[1] - 1
        relative_position_bias_table *= 3.2  # normalize to -3.2, 3.2
        relative_position_bias_table = torch.sign(relative_position_bias_table) * (
            1 - torch.exp(-torch.abs(relative_position_bias_table))
        )
        self.register_buffer(
            "relative_position_bias_table", relative_position_bias_table
        )

        # get pair-wise aligned relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
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
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        self.q1, self.q2 = (
            nn.Linear(dim // 2, dim // 2, bias=True),
            nn.Linear(dim // 2, dim // 2, bias=True),
        )
        self.k1, self.k2 = (
            nn.Linear(dim // 2, dim // 2, bias=True),
            nn.Linear(dim // 2, dim // 2, bias=True),
        )
        self.v1, self.v2 = (
            nn.Linear(dim // 2, dim // 2, bias=True),
            nn.Linear(dim // 2, dim // 2, bias=True),
        )
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj1, self.proj2 = (
            nn.Linear(dim // 2, dim // 2, bias=True),
            nn.Linear(dim // 2, dim // 2, bias=True),
        )
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        x = (
            x.reshape(x.shape[0], x.shape[1], 2, c // 2)
            .permute(2, 0, 1, 3)
            .contiguous()
        )

        # GRL_k
        k = torch.stack((x[0] + self.k1(x[0]), x[1] + self.k2(x[1])), dim=0)
        k = k.permute(1, 2, 0, 3).flatten(2)
        k = (
            k.reshape(b_, n, self.num_heads, c // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        # GRL_q
        q = torch.stack((x[0] + self.q1(x[0]), x[1] + self.q2(x[1])), dim=0)
        q = q.permute(1, 2, 0, 3).flatten(2)
        q = (
            q.reshape(b_, n, self.num_heads, c // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        # GRL_v
        v = torch.stack((x[0] + self.v1(x[0]), x[1] + self.v2(x[1])), dim=0)
        v = v.permute(1, 2, 0, 3).flatten(2)
        v = (
            v.reshape(b_, n, self.num_heads, c // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)

        # send to device
        self.logit_scale = self.logit_scale.to(x.device)
        logit_scale = torch.clamp(
            self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01))
        ).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.ESRPB_MLP(
            self.relative_position_bias_table
        ).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(n, n, -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = x.reshape(b_, n, 2, c // 2).permute(2, 0, 1, 3).contiguous()
        return (
            torch.stack((self.proj1(x[0]), self.proj2(x[1])), dim=0)
            .permute(1, 2, 0, 3)
            .reshape(b_, n, c)
        )


class GRSAB(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=None,
        shift_size=None,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        bi=0,
        img_size=(64, 64),
    ):
        if shift_size is None:
            shift_size = [4, 16]
        if window_size is None:
            window_size = [8, 32]
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.img_size = img_size
        if (bi // 2) % 2 == 1:
            window_size = (window_size[1], window_size[0])
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)

        self.attn = GRSA(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FFN(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        if self.shift_size[0] > 0:
            attn_mask = self.calculate_mask(self.img_size)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate mask for original windows
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (
            slice(0, -self.window_size[0]),
            slice(-self.window_size[0], -self.shift_size[0]),
            slice(-self.shift_size[0], None),
        )
        w_slices = (
            slice(0, -self.window_size[1]),
            slice(-self.window_size[1], -self.shift_size[1]),
            slice(-self.shift_size[1], None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])

        # calculate attention mask
        attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(1)
        return attn_mask.masked_fill(attn_mask != 0, -1e9).masked_fill(
            attn_mask == 0, 0.0
        )

    def forward(self, x, x_size):
        h, w = x_size
        b, _seq_len, c = x.shape
        # assert seq_len == h * w, "input feature has wrong size"

        shortcut = x
        # x = self.norm1(x)
        x = x.view(b, h, w, c)
        # cyclic shift
        if self.shift_size[0] > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)
            )
        else:
            shifted_x = x

        shifted_x = window_partition(shifted_x, self.window_size)
        shifted_x = shifted_x.view(
            shifted_x.shape[0],
            shifted_x.shape[1] * shifted_x.shape[2],
            shifted_x.shape[3],
        )
        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.img_size == x_size:
            attn_windows = self.attn(
                shifted_x, mask=self.attn_mask
            )  # nw*b, window_size*window_size, c
        else:
            attn_windows = self.attn(
                shifted_x, mask=self.calculate_mask(x_size).to(x.device)
            )
        attn_windows = window_reverse(attn_windows, self.window_size, h, w)

        # reverse cyclic shift
        if self.shift_size[0] > 0:
            x = torch.roll(
                attn_windows,
                shifts=(self.shift_size[0], self.shift_size[1]),
                dims=(1, 2),
            )
        else:
            x = attn_windows
        x = x.view(b, h * w, c)
        # FFN
        x = shortcut + self.drop_path(self.norm1(x))
        return x + self.drop_path(self.norm2(self.mlp(x, x_size)))


class GRSAB_Group(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        img_size=(64, 64),
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            GRSAB(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0, 0]
                if (i % 2 == 0)
                else [window_size[0] // 2, window_size[1] // 2],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                bi=i,
                img_size=img_size,
            )
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, x_size):
        for blk in self.blocks:
            x = blk(x, x_size)
        short_cut = x
        x = rearrange(x, "b (h w) c -> b c h w", h=x_size[0], w=x_size[1])
        x = self.conv(x)
        return short_cut + rearrange(x, "b c h w -> b (h w) c")


class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.extend((
            nn.Conv2d(num_feat, scale**2 * num_out_ch, 3, 1, 1),
            nn.PixelShuffle(scale),
        ))
        super().__init__(*m)


@ARCH_REGISTRY.register()
class grformer(nn.Module):
    def __init__(
        self,
        img_size=64,
        in_chans=3,
        embed_dim=60,
        depths=(6, 6, 6, 6),
        num_heads=(3, 3, 3, 3),
        window_size=None,
        mlp_ratio=2,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        upscale=upscale,
        img_range=1.0,
        **kwargs,
    ):
        if window_size is None:
            window_size = [8, 32]
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.5, 0.5, 0.5)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.window_size = window_size
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = GRSAB_Group(
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],  # no impact on SR results
                norm_layer=norm_layer,
                img_size=(img_size, img_size),
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        # for lightweight SR (to save parameters)
        self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
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
        b, c, h, w = x.shape
        x_size = (h, w)
        x = x.flatten(2).transpose(1, 2).contiguous()  # batch seq_len channel
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # batch seq_len channel
        return x.transpose(1, 2).reshape(b, c, h, w)  # batch channel img_h img_w

    def check_image_size(self, x):
        _, _, h, w = x.size()
        max_window_size = max(self.window_size)
        mod_pad_h = (max_window_size - h % max_window_size) % max_window_size
        mod_pad_w = (max_window_size - w % max_window_size) % max_window_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        # for lightweight SR
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.upsample(x)

        x = x / self.img_range + self.mean

        return x[:, :, : H * self.upscale, : W * self.upscale]


@ARCH_REGISTRY.register()
def grformer_medium(**kwargs):
    return grformer(
        img_size=64,
        depths=(6, 6, 6, 6, 6, 6),
        embed_dim=180,
        num_heads=(6, 6, 6, 6, 6, 6),
        **kwargs,
    )


@ARCH_REGISTRY.register()
def grformer_large(**kwargs):
    return grformer(
        img_size=64,
        depths=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        embed_dim=180,
        num_heads=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        **kwargs,
    )
