import torch
from torch import Tensor, nn

from neosr.utils.registry import LOSS_REGISTRY


def rgb_to_luma(img: Tensor) -> Tensor:
    """RGB to CIELAB L*."""
    if not isinstance(img, torch.Tensor):
        msg = f"Input type is not a Tensor. Got {type(img)}"
        raise TypeError(msg)

    if len(img.shape) < 3 or img.shape[-3] != 3:
        msg = f"Input size must have a shape of (*, 3, H, W). Got {img.shape}"
        raise ValueError(msg)

    # permute
    out_img = img.permute(0, 2, 3, 1)

    # linearize rgb
    linear = out_img <= 0.04045
    if torch.any(linear):
        out_img = out_img / 12.92
    else:
        out_img = torch.pow(((out_img + 0.055) / 1.055), 2.4)

    # convert to luma - Y axis of sRGB > XYZ standard
    out_img @= torch.tensor([0.2126, 0.7152, 0.0722])

    # convert Y to L* (from CIELAB L*a*b*)
    # NOTE: will convert from range [0, 1] to range [0,100]
    condition = out_img <= (216 / 24389)
    if torch.any(condition):
        out_img *= 24389 / 27
    else:
        out_img = torch.pow(out_img, (1 / 3)) * 116 - 16

    # normalize to [0, 1] range again
    return torch.clamp((out_img / 100), 0, 1)


@LOSS_REGISTRY.register()
class luma_loss(nn.Module):
    """Luminance Loss.
    Converts images to Y from CIE XYZ and then to CIE L* (from L*a*b*).

    Args:
    ----
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
        super().__init__()
        self.loss_weight = loss_weight
        self.criterion_type = criterion
        self.avgpool = avgpool
        self.scale = scale
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

    def forward(self, net_output: Tensor, gt: Tensor) -> Tensor:
        input_luma = rgb_to_luma(net_output)
        target_luma = rgb_to_luma(gt)

        if self.avgpool:
            input_luma = torch.nn.AvgPool2d(kernel_size=int(self.scale))(input_luma)
            target_luma = torch.nn.AvgPool2d(kernel_size=int(self.scale))(target_luma)

        return self.criterion(input_luma, target_luma) * self.loss_weight
