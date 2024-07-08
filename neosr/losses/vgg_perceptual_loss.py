from collections import OrderedDict

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neosr.archs.arch_util import net_opt
from neosr.archs.vgg_arch import VGGFeatureExtractor
from neosr.utils.registry import LOSS_REGISTRY

upscale, __ = net_opt()


class PatchesKernel3D(nn.Module):
    """Adapted from 'Patch Loss: A Generic Multi-Scale Perceptual Loss for
    Single Image Super-resolution':
    https://www.sciencedirect.com/science/article/pii/S0031320323002108
    https://github.com/Suanmd/Patch-Loss-for-Super-Resolution.
    """

    def __init__(
        self, kernelsize: int, kernelstride: int, kernelpadding: int = 0
    ) -> None:
        super().__init__()
        kernel = (
            torch.eye(kernelsize**2)
            .view(kernelsize**2, 1, kernelsize, kernelsize)
            .clone()
            .detach()
        )
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(kernelsize**2), requires_grad=False)
        self.kernelsize = kernelsize
        self.stride = kernelstride
        self.padding = kernelpadding

    # @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, x: Tensor) -> Tensor:
        batchsize = x.shape[0]
        channels = x.shape[1]
        x = x.reshape(batchsize * channels, x.shape[-2], x.shape[-1]).unsqueeze(1)
        with torch.no_grad():
            x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)

        return (
            x.permute(0, 2, 3, 1)
            .reshape(batchsize, channels, -1, self.kernelsize**2)
            .permute(0, 2, 1, 3)
        )


@LOSS_REGISTRY.register()
class vgg_perceptual_loss(nn.Module):
    """Perceptual loss with VGG19.

    Args:
    ----
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        loss_weight (float): If `loss_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        criterion (str): Criterion used for perceptual loss. Default: 'huber'.
        patchloss (bool): Enables PatchLoss. Default: False.
        ipk (bool): Enables Image Patch Kernel and adds to the final loss. Default: False.
        patch_weight (float): Weight of PatchLoss. Default: 1.0

    """

    def __init__(
        self,
        layer_weights: OrderedDict,
        vgg_type: str = "vgg19",
        use_input_norm: bool = True,
        range_norm: bool = False,
        loss_weight: float = 1.0,
        criterion: str = "huber",
        patchloss: bool = False,
        ipk: bool = False,
        patch_weight: float = 1.0,
        **kwargs,  # noqa: ARG002
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.layer_weights = layer_weights
        self.patch_weights = patch_weight
        self.patchloss = patchloss
        self.ipk = ipk

        if self.patchloss is False and self.ipk is True:
            msg = "Please enable PatchLoss to use IPK."
            raise ValueError(msg)

        if patchloss:
            if upscale == 4:
                self.perceptual_kernels = [4, 8]
                self.ipk_kernels = [7, 11, 15]
            elif upscale == 2:
                self.perceptual_kernels = [3, 6]
                self.ipk_kernels = [3, 5, 7]
            else:
                msg = f"PatchLoss does not support upscale ratio {upscale}."
                raise NotImplementedError(msg)

        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm,
        )

        self.criterion: nn.L1Loss | nn.MSELoss | nn.HuberLoss
        self.criterion_type = criterion
        if self.criterion_type == "l1":
            self.criterion = nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = nn.MSELoss()
        elif self.criterion_type == "huber":
            self.criterion = nn.HuberLoss()
        else:
            msg = f"{criterion} criterion not supported."
            raise NotImplementedError(msg)

    @torch.no_grad()
    # @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def patch(self, x: Tensor, gt: Tensor, is_ipk: bool = False) -> Tensor:
        """Args:
        ----
            is_ipk (bool): defines if it's IPK (Image Patch Kernel)
            or FPK (Feature Patch Kernel). Default: False.

        """
        loss = 0.0

        # IPK
        if is_ipk:
            for _kernel in self.ipk_kernels:
                _patchkernel3d = PatchesKernel3D(_kernel, _kernel // 2).to(
                    x.device, non_blocking=True
                )  # create instance
                x_trans = _patchkernel3d(x)
                gt_trans = _patchkernel3d(gt)
                x_trans = x_trans.reshape(-1, x_trans.shape[-1])
                gt_trans = gt_trans.reshape(-1, gt_trans.shape[-1])
                x_trans = torch.clamp(x_trans, 0.000001, 0.999999)
                gt_trans = torch.clamp(gt_trans, 0.000001, 0.999999)
                dot_x_y = torch.einsum("ik,ik->i", x_trans, gt_trans)

                dy = torch.std(gt_trans, dim=1)
                cosine_x_y = torch.div(
                    torch.div(dot_x_y, torch.sqrt(torch.sum(x_trans**2, dim=1))),
                    torch.sqrt(torch.sum(gt_trans**2, dim=1)),
                )
                cosine_x_y_d = torch.mul((1 - cosine_x_y), dy)  # y = (1-x)dy

        # FPK
        else:
            for _kernel in self.perceptual_kernels:
                _patchkernel3d = PatchesKernel3D(_kernel, _kernel // 2).to(
                    x.device, non_blocking=True
                )  # create instance
                x_trans = _patchkernel3d(x)
                gt_trans = _patchkernel3d(gt)
                x_trans = x_trans.reshape(-1, x_trans.shape[-1])
                gt_trans = gt_trans.reshape(-1, gt_trans.shape[-1])
                dot_x_y = torch.einsum("ik,ik->i", x_trans, gt_trans)

                dy = torch.std(gt_trans, dim=1)
                cosine_x_y = torch.div(
                    torch.div(dot_x_y, torch.sqrt(torch.sum(x_trans**2, dim=1))),
                    torch.sqrt(torch.sum(gt_trans**2, dim=1)),
                )
                cosine_x_y_d = torch.mul((1 - cosine_x_y), dy)  # y = (1-x)dy

        return loss + torch.mean(cosine_x_y_d)

    def forward(self, x: Tensor, gt: Tensor) -> float:
        """Forward function.

        Args:
        ----
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
        -------
            Float: loss value.

        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.loss_weight > 0:
            percep_loss = 0
            for k in x_features:
                if self.patchloss:
                    percep_loss += (
                        self.patch(x_features[k], gt_features[k])
                        * self.layer_weights[k]
                        * self.patch_weights
                        + self.criterion(x_features[k], gt_features[k])
                        * self.layer_weights[k]
                    )
                else:
                    percep_loss += (
                        self.criterion(x_features[k], gt_features[k])
                        * self.layer_weights[k]
                    )

            # add IPK
            if self.patchloss and self.ipk:
                ipk = self.patch(x, gt, is_ipk=True)
                percep_loss += ipk

        return percep_loss * self.loss_weight
