# type: ignore  # noqa: PGH003
import functools

from torch import nn
from torch.nn.utils import spectral_norm

from neosr.utils.registry import ARCH_REGISTRY


def get_conv_layer(
    input_nc, ndf, kernel_size, stride, padding, bias=True, use_sn=False
):
    if not use_sn:
        return nn.Conv2d(
            input_nc,
            ndf,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    return spectral_norm(
        nn.Conv2d(
            input_nc,
            ndf,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    )


@ARCH_REGISTRY.register()
class patchgan(nn.Module):
    """PatchGAN discriminator with spectral normalization, first proposed by:
       https://arxiv.org/abs/1711.11585
       https://arxiv.org/abs/1802.05957

    Args:
    ----
        use_sn (bool): Use spectral_norm or not. If use_sn is True, then norm_type should be none.

    """

    def __init__(
        self,
        num_in_ch=3,
        num_feat=64,
        num_layers=3,
        max_nf_mult=8,
        norm_type="none",
        use_sigmoid=False,
        use_sn=True,
        **kwargs,
    ):
        super(patchgan, self).__init__()

        norm_layer = self._get_norm_layer(norm_type)
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            get_conv_layer(
                num_in_ch,
                num_feat,
                kernel_size=kw,
                stride=2,
                padding=padw,
                use_sn=use_sn,
            ),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, num_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, max_nf_mult)
            sequence += [
                get_conv_layer(
                    num_feat * nf_mult_prev,
                    num_feat * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                    use_sn=use_sn,
                ),
                norm_layer(num_feat * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**num_layers, max_nf_mult)
        sequence += [
            get_conv_layer(
                num_feat * nf_mult_prev,
                num_feat * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
                use_sn=use_sn,
            ),
            norm_layer(num_feat * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        # output 1 channel prediction map
        sequence += [
            get_conv_layer(
                num_feat * nf_mult,
                1,
                kernel_size=kw,
                stride=1,
                padding=padw,
                use_sn=use_sn,
            )
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def _get_norm_layer(self, norm_type="batch"):
        if norm_type == "batch":
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == "instance":
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        elif norm_type == "batchnorm2d":
            norm_layer = nn.BatchNorm2d
        elif norm_type == "none":
            norm_layer = nn.Identity
        else:
            raise NotImplementedError(f"normalization layer [{norm_type}] is not found")

        return norm_layer

    def forward(self, x):
        return self.model(x)
