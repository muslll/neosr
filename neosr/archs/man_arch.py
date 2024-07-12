import torch
import torch.nn.functional as F
from torch import nn

from neosr.archs.arch_util import net_opt
from neosr.utils.registry import ARCH_REGISTRY

upscale, __ = net_opt()


# LKA from VAN (https://github.com/Visual-Attention-Network)
class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 7, padding=7 // 2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 9, stride=1, padding=((9 // 2) * 4), groups=dim, dilation=4
        )
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention(nn.Module):
    def __init__(self, n_feats):
        super().__init__()

        self.norm = LayerNorm(n_feats, data_format="channels_first")
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.proj_1 = nn.Conv2d(n_feats, n_feats, 1)
        self.spatial_gating_unit = LKA(n_feats)
        self.proj_2 = nn.Conv2d(n_feats, n_feats, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(self.norm(x))
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x * self.scale + shorcut
        return x


# ----------------------------------------------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, n_feats):
        super().__init__()

        self.norm = LayerNorm(n_feats, data_format="channels_first")
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        i_feats = 2 * n_feats

        self.fc1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(i_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x * self.scale + shortcut


class CFF(nn.Module):
    def __init__(self, n_feats, drop=0.0, k=2, squeeze_factor=15, attn="GLKA"):
        super().__init__()
        i_feats = n_feats * 2

        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Sequential(
            nn.Conv2d(i_feats, i_feats, 7, 1, 7 // 2, groups=n_feats), nn.GELU()
        )
        self.Conv2 = nn.Conv2d(i_feats, n_feats, 1, 1, 0)

        self.norm = LayerNorm(n_feats, data_format="channels_first")
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()

        # Ghost Expand
        x = self.Conv1(self.norm(x))
        x = self.DWConv1(x)
        x = self.Conv2(x)

        return x * self.scale + shortcut


class SimpleGate(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = n_feats * 2

        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        # self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7//2, groups= n_feats)
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

        self.norm = LayerNorm(n_feats, data_format="channels_first")
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()

        # Ghost Expand
        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * a  # self.DWConv1(a)
        x = self.Conv2(x)

        return x * self.scale + shortcut


# -----------------------------------------------------------------------------------------------------------------
# RCAN-style
class RCBv6(nn.Module):
    def __init__(
        self, n_feats, k, lk=7, res_scale=1.0, style="X", act=nn.SiLU(), deploy=False
    ):
        super().__init__()
        self.LKA = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 5, 1, lk // 2, groups=n_feats),
            nn.Conv2d(
                n_feats, n_feats, 7, stride=1, padding=9, groups=n_feats, dilation=3
            ),
            nn.Conv2d(n_feats, n_feats, 1, 1, 0),
            nn.Sigmoid(),
        )

        # self.LFE2 = LFEv3(n_feats, attn ='CA')

        self.LFE = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
        )

    def forward(self, x, pre_attn=None, RAA=None):
        shortcut = x.clone()
        x = self.LFE(x)

        x = self.LKA(x) * x

        return x + shortcut


# -----------------------------------------------------------------------------------------------------------------


class MLKA_Ablation(nn.Module):
    def __init__(self, n_feats, k=2, squeeze_factor=15):
        super().__init__()
        i_feats = 2 * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats

        self.norm = LayerNorm(n_feats, data_format="channels_first")
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        k = 2

        # Multiscale Large Kernel Attention
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // k, n_feats // k, 7, 1, 7 // 2, groups=n_feats // k),
            nn.Conv2d(
                n_feats // k,
                n_feats // k,
                9,
                stride=1,
                padding=(9 // 2) * 4,
                groups=n_feats // k,
                dilation=4,
            ),
            nn.Conv2d(n_feats // k, n_feats // k, 1, 1, 0),
        )
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // k, n_feats // k, 5, 1, 5 // 2, groups=n_feats // k),
            nn.Conv2d(
                n_feats // k,
                n_feats // k,
                7,
                stride=1,
                padding=(7 // 2) * 3,
                groups=n_feats // k,
                dilation=3,
            ),
            nn.Conv2d(n_feats // k, n_feats // k, 1, 1, 0),
        )
        """self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats//k, n_feats//k, 3, 1, 1, groups= n_feats//k),  
            nn.Conv2d(n_feats//k, n_feats//k, 5, stride=1, padding=(5//2)*2, groups=n_feats//k, dilation=2),
            nn.Conv2d(n_feats//k, n_feats//k, 1, 1, 0))"""

        # self.X3 = nn.Conv2d(n_feats//k, n_feats//k, 3, 1, 1, groups= n_feats//k)
        self.X5 = nn.Conv2d(
            n_feats // k, n_feats // k, 5, 1, 5 // 2, groups=n_feats // k
        )
        self.X7 = nn.Conv2d(
            n_feats // k, n_feats // k, 7, 1, 7 // 2, groups=n_feats // k
        )

        self.proj_first = nn.Sequential(nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(nn.Conv2d(n_feats, n_feats, 1, 1, 0))

    def forward(self, x, pre_attn=None, RAA=None):
        shortcut = x.clone()

        x = self.norm(x)

        x = self.proj_first(x)

        a, x = torch.chunk(x, 2, dim=1)

        # u_1, u_2, u_3= torch.chunk(u, 3, dim=1)
        a_1, a_2 = torch.chunk(a, 2, dim=1)

        a = torch.cat(
            [self.LKA7(a_1) * self.X7(a_1), self.LKA5(a_2) * self.X5(a_2)], dim=1
        )

        x = self.proj_last(x * a) * self.scale + shortcut

        return x


# -----------------------------------------------------------------------------------------------------------------


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
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
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SGAB(nn.Module):
    def __init__(self, n_feats, drop=0.0, k=2, squeeze_factor=15, attn="GLKA"):
        super().__init__()
        i_feats = n_feats * 2

        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

        self.norm = LayerNorm(n_feats, data_format="channels_first")
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()

        # Ghost Expand
        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)

        return x * self.scale + shortcut


class GroupGLKA(nn.Module):
    def __init__(self, n_feats, k=2, squeeze_factor=15):
        super().__init__()
        i_feats = 2 * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats

        self.norm = LayerNorm(n_feats, data_format="channels_first")
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        # Multiscale Large Kernel Attention
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3),
            nn.Conv2d(
                n_feats // 3,
                n_feats // 3,
                9,
                stride=1,
                padding=(9 // 2) * 4,
                groups=n_feats // 3,
                dilation=4,
            ),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0),
        )
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3),
            nn.Conv2d(
                n_feats // 3,
                n_feats // 3,
                7,
                stride=1,
                padding=(7 // 2) * 3,
                groups=n_feats // 3,
                dilation=3,
            ),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0),
        )
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3),
            nn.Conv2d(
                n_feats // 3,
                n_feats // 3,
                5,
                stride=1,
                padding=(5 // 2) * 2,
                groups=n_feats // 3,
                dilation=2,
            ),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0),
        )

        self.X3 = nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3)
        self.X5 = nn.Conv2d(
            n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3
        )
        self.X7 = nn.Conv2d(
            n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3
        )

        self.proj_first = nn.Sequential(nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(nn.Conv2d(n_feats, n_feats, 1, 1, 0))

    def forward(self, x, pre_attn=None, RAA=None):
        shortcut = x.clone()

        x = self.norm(x)

        x = self.proj_first(x)

        a, x = torch.chunk(x, 2, dim=1)

        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)

        a = torch.cat(
            [
                self.LKA3(a_1) * self.X3(a_1),
                self.LKA5(a_2) * self.X5(a_2),
                self.LKA7(a_3) * self.X7(a_3),
            ],
            dim=1,
        )

        x = self.proj_last(x * a) * self.scale + shortcut

        return x


# MAB
class MAB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()

        self.LKA = GroupGLKA(n_feats)

        self.LFE = SGAB(n_feats)

    def forward(self, x, pre_attn=None, RAA=None):
        # large kernel attention
        x = self.LKA(x)

        # local feature extraction
        x = self.LFE(x)

        return x


class LKAT(nn.Module):
    def __init__(self, n_feats):
        super().__init__()

        # self.norm = LayerNorm(n_feats, data_format='channels_first')
        # self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.conv0 = nn.Sequential(nn.Conv2d(n_feats, n_feats, 1, 1, 0), nn.GELU())

        self.att = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats),
            nn.Conv2d(
                n_feats,
                n_feats,
                9,
                stride=1,
                padding=(9 // 2) * 3,
                groups=n_feats,
                dilation=3,
            ),
            nn.Conv2d(n_feats, n_feats, 1, 1, 0),
        )

        self.conv1 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        x = self.conv0(x)
        x = x * self.att(x)
        x = self.conv1(x)
        return x


class ResGroup(nn.Module):
    def __init__(self, n_resblocks, n_feats, res_scale=1.0):
        super(ResGroup, self).__init__()
        self.body = nn.ModuleList([MAB(n_feats) for _ in range(n_resblocks)])

        self.body_t = LKAT(n_feats)

    def forward(self, x):
        res = x.clone()

        for i, block in enumerate(self.body):
            res = block(res)

        x = self.body_t(res) + x

        return x


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range, rgb_mean=(0.5, 0.5, 0.5), rgb_std=(1.0, 1.0, 1.0), sign=-1
    ):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.tensor(rgb_std).to(device="cuda")
        mean = torch.tensor(rgb_mean).to(device="cuda")
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * mean / std
        for p in self.parameters():
            p.requires_grad = False


@ARCH_REGISTRY.register()
class man(nn.Module):
    def __init__(
        self,
        n_resblocks=36,
        n_resgroups=1,
        n_colors=3,
        n_feats=180,
        scale=upscale,
        res_scale=1.0,
    ):
        super(man, self).__init__()

        # res_scale = res_scale
        self.n_resgroups = n_resgroups

        self.sub_mean = MeanShift(1.0)
        self.head = nn.Conv2d(n_colors, n_feats, 3, 1, 1)

        # define body module
        self.body = nn.ModuleList([
            ResGroup(n_resblocks, n_feats, res_scale=res_scale)
            for i in range(n_resgroups)
        ])

        if self.n_resgroups > 1:
            self.body_t = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

        # define tail module
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_colors * (scale**2), 3, 1, 1), nn.PixelShuffle(scale)
        )
        self.add_mean = MeanShift(1.0, sign=1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = x
        for i in self.body:
            res = i(res)
        if self.n_resgroups > 1:
            res = self.body_t(res) + x
        x = self.tail(res)
        x = self.add_mean(x)
        return x


@ARCH_REGISTRY.register()
def man_tiny(**kwargs):
    return man(n_resblocks=5, n_feats=48, **kwargs)


@ARCH_REGISTRY.register()
def man_light(**kwargs):
    return man(n_resblocks=24, n_feats=60, **kwargs)
