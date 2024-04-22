from collections import OrderedDict

import torch
from torch import nn

from neosr.archs.vgg_arch import VGGFeatureExtractor
from neosr.losses.basic_loss import chc
from neosr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with VGG19

    Args:
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
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        criterion (str): Criterion used for perceptual loss. Default: 'huber'.
    """

    def __init__(
        self,
        layer_weights: OrderedDict,
        vgg_type: str = "vgg19",
        use_input_norm: bool = True,
        range_norm: bool = False,
        perceptual_weight: float = 1.0,
        criterion: str = "huber",
        **kwargs,
    ) -> None:
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.layer_weights = layer_weights

        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm,
        )

        self.criterion_type = criterion
        if self.criterion_type == "l1":
            self.criterion = nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = nn.MSELoss()
        elif self.criterion_type == "huber":
            self.criterion = nn.HuberLoss()
        elif self.criterion_type == "fro":
            self.criterion = None
        else:
            raise NotImplementedError(f"{criterion} criterion not supported.")

    def forward(
        self, x: torch.Tensor, gt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == "fro":
                    # note: linalg.norm uses Frobenius norm by default
                    percep_loss += (
                        torch.linalg.norm(x_features[k] - gt_features[k])
                        * self.layer_weights[k]
                    )
                else:
                    percep_loss += (
                        self.criterion(x_features[k], gt_features[k])
                        * self.layer_weights[k]
                    )

            percep_loss *= self.perceptual_weight

        return percep_loss
