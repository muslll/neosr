# type: ignore  # noqa: PGH003
import math

import torch
from torch import nn
from torch.nn import functional as F

from neosr.archs.arch_util import DropPath, net_opt
from neosr.utils.registry import ARCH_REGISTRY

upscale, __ = net_opt()


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_first",
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        assert self.data_format in ["channels_last", "channels_first"]
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


class SADFFM(nn.Module):
    def __init__(
        self, dim: int, expand_ratio: float, bias: bool = True, drop: float = 0.0
    ):
        super().__init__()
        hidden_dims = int(dim * expand_ratio)
        self.linear_in = nn.Conv2d(dim, hidden_dims * 2, kernel_size=1, bias=bias)
        self.SAL = nn.Conv2d(
            hidden_dims * 2,
            hidden_dims * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_dims * 2,
            bias=bias,
        )
        self.linear_out = nn.Conv2d(hidden_dims, dim, kernel_size=1, bias=bias)
        self.DFFM = DFFM(dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.linear_in(x)
        x1, x2 = self.SAL(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.linear_out(x)
        x = self.DFFM(x)
        return self.drop(x)


class DFFM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        act_ratio: float = 0.25,
        act_fn: type[nn.Module] = nn.GELU,
        gate_fn: type[nn.Module] = nn.Sigmoid,
    ):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = LayerNorm(in_channels, data_format="channels_first")
        self.global_reduce = nn.Conv2d(in_channels, reduce_channels, 1)
        self.local_reduce = nn.Conv2d(in_channels, reduce_channels, 1)
        self.act_fn = act_fn()
        self.channel_expand = nn.Conv2d(reduce_channels, in_channels, 1)
        self.spatial_expand = nn.Conv2d(reduce_channels * 2, 1, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        identity = x
        b, _c, _h, _w = x.shape
        x = self.norm(x)
        x_global = self.act_fn(self.global_reduce(F.adaptive_avg_pool2d(x, 1)))
        x_local = self.act_fn(self.local_reduce(x))
        c_attn = self.gate_fn(self.channel_expand(x_global))
        s_attn = self.gate_fn(
            self.spatial_expand(
                torch.cat(
                    [x_local, x_global.expand(b, -1, x.shape[2], x.shape[3])], dim=1
                )
            )
        )
        attn = c_attn * s_attn
        return identity * attn


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features, 3, 1, 1, groups=hidden_features
        )
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Silu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mul_(x.sigmoid())


class MOLRCM(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.split_c1, self.split_c2, self.split_c3 = (
            int((3 / 8) * dim),
            int((1 / 8) * dim),
            int((4 / 8) * dim),
        )
        self.region = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.spatial_1 = nn.Conv2d(
            self.split_c1,
            self.split_c1,
            5,
            stride=1,
            padding=4,
            groups=self.split_c1,
            dilation=2,
        )
        self.spatial_2 = nn.Conv2d(
            self.split_c3,
            self.split_c3,
            7,
            stride=1,
            padding=9,
            groups=self.split_c3,
            dilation=3,
        )
        self.fusion = nn.Conv2d(dim, dim, 1)

        self.gate = Silu()
        self.proj_value = nn.Sequential(nn.Conv2d(dim, dim, 1))
        self.proj_query = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU())
        self.out = nn.Conv2d(dim, dim, 1)  # nn.Identity()

    def forward(self, x_):
        value = self.proj_value(x_)
        query = self.proj_query(x_)
        query = self.region(query)
        query_1 = self.spatial_1(query[:, : self.split_c1, :, :])
        query_2 = query[:, self.split_c1 : (self.split_c1 + self.split_c2), :, :]
        query_3 = self.spatial_2(query[:, (self.split_c1 + self.split_c2) :, :, :])
        out = self.gate(self.fusion(torch.cat([query_1, query_2, query_3], dim=1)))
        return self.out(out * value)


class EIMNBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        bias: bool = True,
        drop: float = 0.0,
        drop_path: float = 0.0,
        norm: type[nn.Module] = nn.BatchNorm2d,
    ):
        super().__init__()
        self.norm1 = norm(dim)
        self.attn = MOLRCM(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm(dim)
        self.mlp = SADFFM(dim, mlp_ratio, bias, drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True
        )
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True
        )

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x))
        )
        return x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
        )


@ARCH_REGISTRY.register()
class eimn(nn.Module):
    def __init__(
        self,
        embed_dims: int = 64,
        scale: int = upscale,
        depths: int = 1,
        mlp_ratios: float = 2.66,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        num_stages: int = 16,
        freeze_param: bool = False,
        norm: type[nn.Module] = nn.BatchNorm2d,
    ):
        super().__init__()
        depths_ = [depths] * num_stages

        self.num_stages = num_stages

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_))]
        cur = 0

        self.head = nn.Sequential(nn.Conv2d(3, embed_dims, 3, 1, 1))
        self.tail = nn.Sequential(
            nn.Conv2d(embed_dims, 3 * scale * scale, 3, 1, 1), nn.PixelShuffle(scale)
        )

        for i in range(self.num_stages):
            block = nn.ModuleList([
                EIMNBlock(
                    dim=embed_dims,
                    mlp_ratio=mlp_ratios,
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    norm=nn.BatchNorm2d,
                )
                for j in range(depths_[i])
            ])
            norm = nn.LayerNorm(embed_dims)
            cur += depths_[i]

            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        if freeze_param:
            self.freeze_para()

    def freeze_param(self):
        for name, param in self.named_parameters():
            # print(name, param.shape)
            if name.split(".")[0] == "tail":
                param.requires_grad = True
            else:
                param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m, std=0.02, bias=0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m, val=1.0, bias=0.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def forward(self, x):
        x = self.head(x)
        identity = x

        for i in range(self.num_stages):
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            for blk in block:
                x = blk(x)
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            x = norm(x)
            x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        return self.tail(identity + x)


@ARCH_REGISTRY.register()
def eimn_l(scale: int = 2) -> eimn:
    return eimn(embed_dims=64, num_stages=16)


@ARCH_REGISTRY.register()
def eimn_a(scale: int = 2) -> eimn:
    return eimn(embed_dims=64, num_stages=14)
