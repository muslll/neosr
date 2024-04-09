import torch
from torch import nn
from torch.nn import functional as F

from neosr.losses.loss_util import weighted_loss
from neosr.utils.registry import LOSS_REGISTRY

_reduction_modes = ["none", "mean", "sum"]


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction="none")


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction="none")


@weighted_loss
def huber_loss(
    pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=1.0)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(L1Loss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction
        )


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(MSELoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction
        )


@LOSS_REGISTRY.register()
class HuberLoss(nn.Module):
    """HuberLoss

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        delta (float): Specifies the threshold at which to change between
            delta-scaled L1 and L2 loss. The value must be positive. Default: 1.0
    """

    def __init__(
        self, loss_weight: float = 1.0, reduction: str = "mean", delta: float = 1.0
    ) -> None:
        super(HuberLoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.delta = delta

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, weight: float = None, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """

        return self.loss_weight * huber_loss(
            pred, target, weight, delta=self.delta, reduction=self.reduction
        )


@LOSS_REGISTRY.register()
class chc(nn.Module):
    """Clipped pseudo-Huber with Cosine Similarity Loss

       For reference on research, see:
       https://github.com/HolmesShuan/AIM2020-Real-Super-Resolution
       https://github.com/dmarnerides/hdr-expandnet

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        criterion (str): Specifies the loss to apply.
            Supported choices are 'l1' and 'huber'. Default: 'huber'.
        loss_lambda (float):  constant factor that adjusts the contribution of the cosine similarity term
        clip_min (float): threshold that sets the gradients of well-trained pixels to zeros
        clip_max (float): max clip limit, can act as a noise filter
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction: str = "mean",
        criterion: str = "huber",
        loss_lambda: float = 0,
        clip_min: float = 0.003921,
        clip_max: float = 0.996078,
    ) -> None:
        super(chc, self).__init__()

        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        # Loss params
        self.loss_weight = loss_weight
        self.criterion = criterion

        # CoSim
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-20)
        self.loss_lambda = loss_lambda  # 5/255 = 0.019607

        # Clip
        self.clip_min = clip_min  # 1/255 = 0.03921
        self.clip_max = clip_max

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        cosine_term = (1 - self.similarity(pred, target)).mean()

        # absolute mean
        if self.criterion == "l1":
            loss = torch.mean(
                torch.clamp(
                    (torch.abs(pred - target) + self.loss_lambda * cosine_term),
                    self.clip_min,
                    self.clip_max,
                )
            )
        # pseudo-huber (charbonnier)
        elif self.criterion == "huber":
            loss = torch.mean(
                torch.clamp(
                    (
                        torch.sqrt((pred - target) ** 2 + 1e-12)
                        + self.loss_lambda * cosine_term
                    ),
                    self.clip_min,
                    self.clip_max,
                )
            )
        else:
            raise NotImplementedError(f"{self.criterion} not implemented.")

        return self.loss_weight * loss
