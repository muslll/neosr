import torch
from torch import Tensor, nn

from neosr.losses.basic_loss import chc
from neosr.utils.registry import LOSS_REGISTRY


def rgb_to_cbcr(img: Tensor) -> Tensor:
    """RGB to *CbCr. Outputs tensor with only CbCr channels.
    ITU-R BT.601 primaries are used in this converison.

    Args:
    ----
        img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.

    Returns:
    -------
        (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.

    """
    if not isinstance(img, torch.Tensor):
        msg = f"Input type is not a Tensor. Got {type(img)}"
        raise TypeError(msg)

    if len(img.shape) < 3 or img.shape[-3] != 3:
        msg = f"Input size must have a shape of (*, 3, H, W). Got {img.shape}"
        raise ValueError(msg)

    # bt.601 matrices in 16-240 range
    weight = torch.tensor([
        [65.481, -37.797, 112.0],
        [128.553, -74.203, -93.786],
        [24.966, 112.0, -18.214],
    ]).to(img)
    # limited to full range
    bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
    out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias
    # 0-1 normalization
    out_img /= 255.0
    # CbCr-only
    return out_img[:, 1:, :, :]


@LOSS_REGISTRY.register()
class color_loss(nn.Module):
    """Color Consistency Loss.
    Converts images to chroma-only and compares both.

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

        self.criterion: nn.L1Loss | nn.MSELoss | nn.HuberLoss | chc

        if self.criterion_type == "l1":
            self.criterion = nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = nn.MSELoss()
        elif self.criterion_type == "huber":
            self.criterion = nn.HuberLoss()
        elif self.criterion_type == "chc":
            self.criterion = chc()
        else:
            msg = f"{criterion} criterion has not been supported."
            raise NotImplementedError(msg)

    def forward(self, net_output: Tensor, gt: Tensor) -> Tensor:
        input_uv = rgb_to_cbcr(net_output)
        target_uv = rgb_to_cbcr(gt)

        # TODO: test downscale operation
        if self.avgpool:
            input_uv = torch.nn.AvgPool2d(kernel_size=int(self.scale))(input_uv)
            target_uv = torch.nn.AvgPool2d(kernel_size=int(self.scale))(target_uv)

        return self.criterion(input_uv, target_uv) * self.loss_weight
