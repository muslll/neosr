import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from neosr.utils.registry import ARCH_REGISTRY
from .arch_util import net_opt

upscale, training = net_opt()


def flow_warp(
    x, flow, interp_mode="bilinear", padding_mode="zeros", align_corners=True
):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x), indexing="ij"
    )
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    return output


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

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
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0.0, requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0.0)


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    if len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    raise NotImplementedError


def batch_index_fill(x, x1, x2, idx1, idx2):
    B, N, C = x.size()
    B, N1, C = x1.size()
    B, N2, C = x2.size()

    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N
    idx2 = idx2 + offset * N

    x = x.reshape(B * N, C)

    x[idx1.reshape(-1)] = x1.reshape(B * N1, C)
    x[idx2.reshape(-1)] = x2.reshape(B * N2, C)

    x = x.reshape(B, N, C)
    return x


class PredictorLG(nn.Module):
    """Importance Score Predictor"""

    def __init__(self, dim, window_size=8, k=4, ratio=0.5):
        super().__init__()

        self.ratio = ratio
        self.window_size = window_size
        cdim = dim + k
        embed_dim = window_size**2
        self.training = training

        self.in_conv = nn.Sequential(
            nn.Conv2d(cdim, cdim // 4, 1),
            LayerNorm(cdim // 4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.out_offsets = nn.Sequential(
            nn.Conv2d(cdim // 4, cdim // 8, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(cdim // 8, 2, 1),
        )

        self.out_mask = nn.Sequential(
            nn.Linear(embed_dim, window_size),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(window_size, 2),
            nn.Softmax(dim=-1),
        )

        self.out_CA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(cdim // 4, dim, 1), nn.Sigmoid()
        )

        self.out_SA = nn.Sequential(nn.Conv2d(cdim // 4, 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, input_x, mask=None, ratio=0.5, train_mode=False):
        x = self.in_conv(input_x)

        offsets = self.out_offsets(x)
        offsets = offsets.tanh().mul(8.0)

        ca = self.out_CA(x)
        sa = self.out_SA(x)

        x = torch.mean(x, keepdim=True, dim=1)

        x = rearrange(
            x,
            "b c (h dh) (w dw) -> b (h w) (dh dw c)",
            dh=self.window_size,
            dw=self.window_size,
        )
        B, N, C = x.size()

        pred_score = self.out_mask(x)
        mask = F.gumbel_softmax(pred_score, hard=True, dim=2)[:, :, 0:1]

        if self.training or train_mode:
            return mask, offsets, ca, sa
        score = pred_score[:, :, 0]
        B, N = score.shape
        r = torch.mean(mask, dim=(0, 1)) * 1.0
        if self.ratio == 1:
            num_keep_node = N  # int(N * r) #int(N * r)
        else:
            num_keep_node = min(int(N * r * 2 * self.ratio), N)
        idx = torch.argsort(score, dim=1, descending=True)
        idx1 = idx[:, :num_keep_node]
        idx2 = idx[:, num_keep_node:]
        return [idx1, idx2], offsets, ca, sa


class CAMixer(nn.Module):
    def __init__(self, dim, window_size=8, bias=True, is_deformable=True, ratio=0.5):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.is_deformable = is_deformable
        self.ratio = ratio
        self.training = training

        k = 3
        d = 2

        self.project_v = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)
        self.project_q = nn.Linear(dim, dim, bias=bias)
        self.project_k = nn.Linear(dim, dim, bias=bias)

        # Conv
        self.conv_sptial = nn.Sequential(
            nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim),
            nn.Conv2d(
                dim, dim, k, stride=1, padding=((k // 2) * d), groups=dim, dilation=d
            ),
        )
        self.project_out = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)

        self.act = nn.GELU()
        # Predictor
        self.route = PredictorLG(dim, window_size, ratio=ratio)

    def forward(self, x, condition_global=None, mask=None, train_mode=False):
        N, C, H, W = x.shape

        v = self.project_v(x)

        if self.is_deformable:
            condition_wind = (
                torch.stack(
                    torch.meshgrid(
                        torch.linspace(-1, 1, self.window_size),
                        torch.linspace(-1, 1, self.window_size),
                        indexing="ij",
                    )
                )
                .type_as(x)
                .unsqueeze(0)
                .repeat(N, 1, H // self.window_size, W // self.window_size)
            )
            if condition_global is None:
                _condition = torch.cat([v, condition_wind], dim=1)
            else:
                _condition = torch.cat([v, condition_global, condition_wind], dim=1)

        mask, offsets, ca, sa = self.route(
            _condition, ratio=self.ratio, train_mode=train_mode
        )

        q = x
        k = x + flow_warp(
            x,
            offsets.permute(0, 2, 3, 1),
            interp_mode="bilinear",
            padding_mode="border",
        )
        qk = torch.cat([q, k], dim=1)

        vs = v * sa

        v = rearrange(
            v,
            "b c (h dh) (w dw) -> b (h w) (dh dw c)",
            dh=self.window_size,
            dw=self.window_size,
        )
        vs = rearrange(
            vs,
            "b c (h dh) (w dw) -> b (h w) (dh dw c)",
            dh=self.window_size,
            dw=self.window_size,
        )
        qk = rearrange(
            qk,
            "b c (h dh) (w dw) -> b (h w) (dh dw c)",
            dh=self.window_size,
            dw=self.window_size,
        )

        if self.training or train_mode:
            N_ = v.shape[1]
            v1, v2 = v * mask, vs * (1 - mask)
            qk1 = qk * mask
        else:
            idx1, idx2 = mask
            _, N_ = idx1.shape
            v1, v2 = batch_index_select(v, idx1), batch_index_select(vs, idx2)
            qk1 = batch_index_select(qk, idx1)

        v1 = rearrange(
            v1,
            "b n (dh dw c) -> (b n) (dh dw) c",
            n=N_,
            dh=self.window_size,
            dw=self.window_size,
        )
        qk1 = rearrange(
            qk1,
            "b n (dh dw c) -> b (n dh dw) c",
            n=N_,
            dh=self.window_size,
            dw=self.window_size,
        )

        q1, k1 = torch.chunk(qk1, 2, dim=2)
        q1 = self.project_q(q1)
        k1 = self.project_k(k1)
        q1 = rearrange(
            q1,
            "b (n dh dw) c -> (b n) (dh dw) c",
            n=N_,
            dh=self.window_size,
            dw=self.window_size,
        )
        k1 = rearrange(
            k1,
            "b (n dh dw) c -> (b n) (dh dw) c",
            n=N_,
            dh=self.window_size,
            dw=self.window_size,
        )

        attn = q1 @ k1.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        f_attn = attn @ v1

        f_attn = rearrange(
            f_attn,
            "(b n) (dh dw) c -> b n (dh dw c)",
            b=N,
            n=N_,
            dh=self.window_size,
            dw=self.window_size,
        )

        if not (self.training or train_mode):
            attn_out = batch_index_fill(v.clone(), f_attn, v2.clone(), idx1, idx2)
        else:
            attn_out = f_attn + v2

        attn_out = rearrange(
            attn_out,
            "b (h w) (dh dw c) -> b (c) (h dh) (w dw)",
            h=H // self.window_size,
            w=W // self.window_size,
            dh=self.window_size,
            dw=self.window_size,
        )

        out = attn_out
        out = self.act(self.conv_sptial(out)) * ca + out
        out = self.project_out(out)

        if self.training:
            return out, torch.mean(mask, dim=1)
        return out


class GatedFeedForward(nn.Module):
    def __init__(self, dim, mult=1, bias=False, dropout=0.0):
        super().__init__()
        self.dim = dim

        self.project_in = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            dim * 2,
            dim * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Block(nn.Module):
    def __init__(self, n_feats, window_size=8, ratio=0.5):
        super(Block, self).__init__()

        self.n_feats = n_feats
        self.norm1 = LayerNorm(n_feats)
        self.mixer = CAMixer(n_feats, window_size=window_size, ratio=ratio)
        self.norm2 = LayerNorm(n_feats)
        self.ffn = GatedFeedForward(n_feats)
        self.training = training

    def forward(self, x, condition_global=None):
        if self.training:
            res, decision = self.mixer(x, condition_global)
            x = self.norm1(x + res)
            res = self.ffn(x)
            x = self.norm2(x + res)
            return x, decision
        res = self.mixer(x, condition_global)
        x = self.norm1(x + res)
        res = self.ffn(x)
        x = self.norm2(x + res)
        return x


class Group(nn.Module):
    def __init__(self, n_feats, n_block, window_size=8, ratio=0.5):
        super(Group, self).__init__()

        self.n_feats = n_feats

        self.body = nn.ModuleList([
            Block(n_feats, window_size=window_size, ratio=ratio) for i in range(n_block)
        ])
        self.body_tail = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        self.training = training

    def forward(self, x, condition_global=None):
        decision = []
        shortcut = x.clone()
        if self.training:
            for _, blk in enumerate(self.body):
                x, mask = blk(x, condition_global)
                decision.append(mask)
            x = self.body_tail(x) + shortcut
            return x, decision
        for _, blk in enumerate(self.body):
            x = blk(x, condition_global)
        x = self.body_tail(x) + shortcut
        return x


@ARCH_REGISTRY.register()
class camixersr(nn.Module):
    def __init__(
        self,
        n_block=[4, 4, 6, 6],
        n_group=4,
        n_colors=3,
        n_feats=60,
        scale=upscale,
        ratio=0.5,
        tile=None,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.window_sizes = 16
        self.tile = tile

        self.head = nn.Conv2d(n_colors, n_feats, 3, 1, 1)

        self.global_predictor = nn.Sequential(
            nn.Conv2d(n_feats, 8, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(8, 2, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.scale = scale
        # define body module
        self.body = nn.ModuleList([
            Group(
                n_feats, n_block=n_block[i], window_size=self.window_sizes, ratio=ratio
            )
            for i in range(n_group)
        ])
        self.body_tail = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        # define tail module
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_colors * (scale**2), 3, 1, 1), nn.PixelShuffle(scale)
        )

        self.training = training
        self.ratio = ratio

    def forward_origin(self, x):
        decision = []
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        x = self.head(x)

        condition_global = self.global_predictor(x)
        shortcut = x.clone()
        if self.training:
            for _, blk in enumerate(self.body):
                x, mask = blk(x, condition_global)
                decision.extend(mask)
        else:
            for _, blk in enumerate(self.body):
                x = blk(x, condition_global)

        x = self.body_tail(x)
        x = x + shortcut
        x = self.tail(x)

        if self.training:
            return x[
                :, :, 0 : H * self.scale, 0 : W * self.scale
            ]  # ,  2*self.ratio*(torch.mean(torch.cat(decision,dim=1),dim=(0,1))-0.5)**2
        return x[:, :, 0 : H * self.scale, 0 : W * self.scale]

    def forward(self, img_lq, tile=None):
        tile = self.tile
        if tile is None or self.training:
            # test the image as a whole
            output = self.forward_origin(img_lq)
        else:
            # test the image tile by tile or use TileModel
            b, c, h, w = img_lq.size()
            tile = min(tile, h, w)
            tile_overlap = tile // 16
            sf = self.scale

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
            E = torch.zeros(b, c, h * sf, w * sf).type_as(img_lq)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = img_lq[..., h_idx : h_idx + tile, w_idx : w_idx + tile]
                    out_patch = self.forward_origin(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[
                        ...,
                        h_idx * sf : (h_idx + tile) * sf,
                        w_idx * sf : (w_idx + tile) * sf,
                    ].add_(out_patch)
                    W[
                        ...,
                        h_idx * sf : (h_idx + tile) * sf,
                        w_idx * sf : (w_idx + tile) * sf,
                    ].add_(out_patch_mask)
            output = E.div_(W)

        return output

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_sizes
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x
