import torch
import torch.nn.functional as F
from torch import nn

from neosr.utils.registry import ARCH_REGISTRY
from .arch_util import net_opt

upscale, training = net_opt()


class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0, bias=True):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim,hidden_dim, 3, 1, 1, groups=dim, bias=bias),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=bias)
        )
        self.act =nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=bias)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x

# Partial-convolution Feed-forward Network
class PCFN(nn.Module):
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25, bias=True):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        self.conv_0 = nn.Conv2d(dim,hidden_dim, 1, 1, 0, bias=bias)
        self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1, bias=bias)

        self.act =nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=bias)

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim
        self.training = training

    def forward(self, x):
        if self.training:
            x = self.act(self.conv_0(x))
            x1, x2 = torch.split(x,[self.p_dim,self.hidden_dim-self.p_dim],dim=1)
            x1 = self.act(self.conv_1(x1))
            x = self.conv_2(torch.cat([x1,x2], dim=1))
        else:
            x = self.act(self.conv_0(x))
            x[:,:self.p_dim,:,:] = self.act(self.conv_1(x[:,:self.p_dim,:,:]))
            x = self.conv_2(x)
        return x

# Self-Modulation Feature Aggregation Moule
class SMFA(nn.Module):
    def __init__(self, dim=36, bias=True):
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim*2, 1, 1, 0, bias=bias)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)

        self.lde = DMlp(dim, 2, bias)

        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=bias)

        self.gelu = nn.GELU()
        self.down_scale = 8

    def forward(self, f):
        _,_,h,w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2,-1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s +  x_v  )), size=(h,w), mode='nearest')
        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)


# Feature Modulation Block
class FMB(nn.Module):
    def __init__(self, dim, ffn_scale=2.0, bias=True):
        super().__init__()

        self.smfa = SMFA(dim, bias)
        self.pcfn = PCFN(dim, ffn_scale, bias=bias)

    def forward(self, x):
        x = self.smfa(F.normalize(x)) + x
        x = self.pcfn(F.normalize(x)) + x
        return x

@ARCH_REGISTRY.register()
class smfan(nn.Module):
    def __init__(self, dim=24, n_blocks=8, ffn_scale=1.5, upscaling_factor=upscale, bias=False):
        super().__init__()
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1, bias=bias)

        self.feats = nn.Sequential(*[FMB(dim, ffn_scale) for _ in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1, bias=bias),
            nn.PixelShuffle(upscaling_factor)
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        return self.to_img(x) 
