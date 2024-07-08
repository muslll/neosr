import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neosr.losses.basic_loss import chc
from neosr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class gw_loss(nn.Module):
    """Gradient-Weighted loss, adapted from 'Component Divide-and-Conquer for Real-World
    Image Super-Resolution': https://arxiv.org/abs/2008.01928.
    """

    def __init__(
        self,
        corner: bool = True,
        criterion: str | None = None,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.corner = corner
        self.loss_weight = loss_weight
        self.criterion_type = criterion
        self.criterion: nn.L1Loss | nn.MSELoss | nn.HuberLoss | chc | None

        if self.criterion_type == "l1":
            self.criterion = nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = nn.MSELoss()
        elif self.criterion_type == "huber":
            self.criterion = nn.HuberLoss()
        elif self.criterion_type == "chc":
            self.criterion = chc()
        elif self.criterion_type is None:
            self.criterion = None
        else:
            msg = f"{criterion} criterion has not been supported."
            raise NotImplementedError(msg)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        Y_x1 = torch.mean(x1, dim=1, keepdim=True)
        Y_x2 = torch.mean(x2, dim=1, keepdim=True)
        sobel_0 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_90 = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        sobel_45 = torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
        sobel_135 = torch.tensor([[0, -1, -2], [1, 0, -1], [2, 1, 0]])

        _b, c, _w, _h = Y_x1.shape
        sobel_0 = sobel_0.expand(c, 1, 3, 3)
        sobel_90 = sobel_90.expand(c, 1, 3, 3)
        sobel_45 = sobel_45.expand(c, 1, 3, 3)
        sobel_135 = sobel_135.expand(c, 1, 3, 3)
        sobel_0 = sobel_0.type_as(Y_x1)
        sobel_90 = sobel_90.type_as(Y_x1)
        sobel_45 = sobel_0.type_as(Y_x1)
        sobel_135 = sobel_90.type_as(Y_x1)

        with torch.no_grad():
            weight_0 = nn.Parameter(data=sobel_0)
            weight_90 = nn.Parameter(data=sobel_90)
            weight_45 = nn.Parameter(data=sobel_45)
            weight_135 = nn.Parameter(data=sobel_135)

        I1_0 = F.conv2d(Y_x1, weight_0, stride=1, padding=1, groups=c)
        I2_0 = F.conv2d(Y_x2, weight_0, stride=1, padding=1, groups=c)
        I1_90 = F.conv2d(Y_x1, weight_90, stride=1, padding=1, groups=c)
        I2_90 = F.conv2d(Y_x2, weight_90, stride=1, padding=1, groups=c)
        I1_45 = F.conv2d(Y_x1, weight_45, stride=1, padding=1, groups=c)
        I2_45 = F.conv2d(Y_x2, weight_45, stride=1, padding=1, groups=c)
        I1_135 = F.conv2d(Y_x1, weight_135, stride=1, padding=1, groups=c)
        I2_135 = F.conv2d(Y_x2, weight_135, stride=1, padding=1, groups=c)
        d0 = torch.abs(I1_0 - I2_0)
        d90 = torch.abs(I1_90 - I2_90)
        d45 = torch.abs(I1_45 - I2_45)
        d135 = torch.abs(I1_135 - I2_135)

        if self.criterion is not None:
            reduction = self.criterion(x1, x2)
        else:
            reduction = torch.abs(x1 - x2)

        if self.corner:
            d0 = d0.expand(x1.shape)
            d90 = d90.expand(x1.shape)
            d45 = d45.expand(x1.shape)
            d135 = d135.expand(x1.shape)
            loss = (
                (1 + 4 * d0)
                * (1 + 4 * d90)
                * (1 + 4 * d45)
                * (1 + 4 * d135)
                * reduction
            )

        else:
            d = torch.cat((d0, d90, d45, d135), dim=1)
            d = torch.max(d, dim=1, keepdim=True)[0]
            d = d.expand(x1.shape)
            loss = (1 + 4 * d) * reduction

        return torch.mean(loss) * self.loss_weight
