import torch
from torch import nn

from neosr.losses.basic_loss import chc
from neosr.utils.color_util import rgb_to_cbcr
from neosr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class colorloss(nn.Module):
    """Color Consistency Loss.
    Converts images to chroma-only and compares both.

    Args:
        criterion (str): loss type. Default: 'huber'
        avgpool (bool): apply downscaling after conversion. Default: False
        scale (int): value used by avgpool. Default: 4
        loss_weight (float): weight for colorloss. Default: 1.0
    """

    def __init__(
        self,
        criterion: str = "huber",
        avgpool: bool = False,
        scale: int = 2,
        loss_weight: float = 1.0,
    ) -> None:
        super(colorloss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion_type = criterion
        self.avgpool = avgpool
        self.scale = scale

        if self.criterion_type == "l1":
            self.criterion = nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = nn.MSELoss()
        elif self.criterion_type == "huber":
            self.criterion = nn.HuberLoss()
        elif self.criterion_type == "chc":
            self.criterion = chc()
        else:
            raise NotImplementedError(f"{criterion} criterion has not been supported.")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_uv = rgb_to_cbcr(input)
        target_uv = rgb_to_cbcr(target)

        # TODO: test downscale operation
        if self.avgpool:
            input_uv = torch.nn.AvgPool2d(kernel_size=int(self.scale))(input_uv)
            target_uv = torch.nn.AvgPool2d(kernel_size=int(self.scale))(target_uv)

        return self.criterion(input_uv, target_uv) * self.loss_weight
