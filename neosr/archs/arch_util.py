import collections.abc
import functools
import inspect
import math
import warnings
from collections.abc import Mapping
from itertools import repeat
from pathlib import Path
from typing import Any, Protocol, TypeVar

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from neosr.utils.options import parse_options


def net_opt():
    # initialize options parsing
    root_path = Path(__file__).parents[2]
    opt, args = parse_options(root_path, is_train=True)

    # set variable for scale factor and training phase
    # conditions needed due to convert.py

    if args.input is None:
        upscale = opt["scale"]
        if "train" in opt["datasets"]:
            training = True
        else:
            training = False
    else:
        upscale = args.scale
        training = False

    return upscale, training


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d | nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


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
                f"scale {scale} is not supported. Supported scales: 2^n and 3."
            )
        super(Upsample, self).__init__(*m)


######################
# DySample
######################
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels: int, scale: int = 2, style: str = 'lp', groups: int = 4, dyscope: bool = False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        self.end_conv = nn.Conv2d(in_channels, in_channels // scale ** 2, kernel_size=1)
        assert style in ['lp', 'pl']

        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            assert in_channels >= groups and in_channels % groups == 0
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)
            self.offset_forward = self.offset_scope_pl if style == 'pl' else self.offset_scope_lp
        else:
            self.offset_forward = self.offset_no_scope_pl if style == 'pl' else self.offset_no_scope_lp

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def offset_scope_lp(self, x):
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        return self.sample(x, offset)

    def offset_no_scope_lp(self, x):
        offset = self.offset(x) * 0.25 + self.init_pos

        return self.sample(x, offset)

    def offset_scope_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        return self.sample(x, offset)

    def offset_no_scope_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        return self.end_conv(self.offset_forward(x))

# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)

    """
    # old implementation
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    """

    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)

    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        __, training = net_opt()
        self.training = training

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def store_neosr_defaults(*, extra_parameters: Mapping[str, object] = {}):
    """
    Stores the neosr default hyperparameters in a `neosr_params` attribute.
    Based on Spandrel implementation (MIT license):
        https://github.com/chaiNNer-org/spandrel
    """

    def get_arg_defaults(spec: inspect.FullArgSpec) -> dict[str, Any]:
        defaults = {}
        if spec.kwonlydefaults is not None:
            defaults = spec.kwonlydefaults

        if spec.defaults is not None:
            defaults = {
                **defaults,
                **dict(
                    zip(spec.args[-len(spec.defaults) :], spec.defaults, strict=False)
                ),
            }

        return defaults

    class WithHyperparameters(Protocol):
        neosr_params: dict[str, Any]

    C = TypeVar("C", bound=WithHyperparameters)

    def inner(cls: type[C]) -> type[C]:
        old_init = cls.__init__

        spec = inspect.getfullargspec(old_init)
        defaults = get_arg_defaults(spec)

        @functools.wraps(old_init)
        def new_init(self: C, **kwargs):
            # remove extra parameters from kwargs
            for k, v in extra_parameters.items():
                if k in kwargs:
                    if kwargs[k] != v:
                        raise ValueError(
                            f"Expected hyperparameter {k} to be {v}, but got {kwargs[k]}"
                        )
                    del kwargs[k]

            self.hyperparameters = {**extra_parameters, **defaults, **kwargs}
            old_init(self, **kwargs)

        cls.__init__ = new_init
        return cls

    return inner


# From PyTorch
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
