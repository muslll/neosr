import torch
from torch import nn

from neosr.utils.color_util import rgb_to_luma
from neosr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class lumaloss(nn.Module):
    """Luminance Loss.
    Converts images to Y from CIE XYZ and then to CIE L* (from L*a*b*)

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
        super(lumaloss, self).__init__()
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
        else:
            raise NotImplementedError(f"{criterion} criterion has not been supported.")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_luma = rgb_to_luma(input)
        target_luma = rgb_to_luma(target)

        if self.avgpool:
            input_luma = torch.nn.AvgPool2d(kernel_size=int(self.scale))(input_luma)
            target_luma = torch.nn.AvgPool2d(kernel_size=int(self.scale))(target_luma)

        return self.criterion(input_luma, target_luma) * self.loss_weight
