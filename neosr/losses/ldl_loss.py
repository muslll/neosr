import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neosr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class ldl_loss(nn.Module):
    """LDL loss. Adapted from 'Details or Artifacts: A Locally Discriminative
    Learning Approach to Realistic Image Super-Resolution':
    https://arxiv.org/abs/2203.09195.

    Args:
    ----
        criterion (str): loss type. Default: 'huber'
        loss_weight (float): weight for colorloss. Default: 1.0
        ksize (int): size of the local window. Default: 7

    """

    def __init__(
        self, criterion: str = "huber", loss_weight: float = 1.0, ksize: int = 7
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.ksize = ksize
        self.criterion_type = criterion
        self.criterion: nn.L1Loss | nn.MSELoss | nn.HuberLoss

        if self.criterion_type == "l1":
            self.criterion = nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = nn.MSELoss()
        elif self.criterion_type == "huber":
            self.criterion = nn.HuberLoss()
        else:
            msg = f"{criterion} criterion has not been supported."
            raise NotImplementedError(msg)

    def get_local_weights(self, residual: Tensor) -> Tensor:
        """Get local weights for generating the artifact map of LDL.

        It is only called by the `get_refined_artifact_map` function.

        Args:
        ----
            residual (Tensor): Residual between predicted and ground truth images.

        Returns:
        -------
            Tensor: weight for each pixel to be discriminated as an artifact pixel

        """
        pad = (self.ksize - 1) // 2
        residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode="reflect")

        unfolded_residual = residual_pad.unfold(2, self.ksize, 1).unfold(
            3, self.ksize, 1
        )
        return (
            torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True)
            .squeeze(-1)
            .squeeze(-1)
        )

    def get_refined_artifact_map(self, img_gt: Tensor, img_output: Tensor) -> Tensor:
        """Calculate the artifact map of LDL
        (Details or Artifacts: A Locally Discriminative Learning Approach to Realistic Image Super-Resolution. In CVPR 2022).

        Args:
        ----
            img_gt (Tensor): ground truth images.
            img_output (Tensor): output images given by the optimizing model.

        Returns:
        -------
            overall_weight: weight for each pixel to be discriminated as an artifact pixel
            (calculated based on both local and global observations).

        """
        residual_sr = torch.sum(torch.abs(img_gt - img_output), 1, keepdim=True)

        patch_level_weight = torch.var(
            residual_sr.clone(), dim=(-1, -2, -3), keepdim=True
        ) ** (1 / 5)
        pixel_level_weight = self.get_local_weights(residual_sr.clone())
        return patch_level_weight * pixel_level_weight

    def forward(self, net_output: Tensor, gt: Tensor) -> Tensor:
        overall_weight = self.get_refined_artifact_map(gt, net_output)
        self.output = torch.mul(overall_weight, net_output)
        self.gt = torch.mul(overall_weight, gt)

        return self.criterion(self.output, self.gt) * self.loss_weight
