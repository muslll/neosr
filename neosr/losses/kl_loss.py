import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neosr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class kl_loss(nn.Module):
    """KL-Divergence loss.

    Args:
    ----
        loss_weight (float): weight for the loss. Default: 1.0
    """

    def __init__(self, loss_weight: float = 1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, net_output: Tensor, gt: Tensor):
        # Convert net_output and gt to probability distributions
        net_output_prob = F.softmax(net_output, dim=1)
        gt_prob = F.softmax(gt, dim=1)
        # Compute log probabilities
        net_output_log_prob = torch.log(net_output_prob + 1e-8)
        # Compute KL divergence
        loss = F.kl_div(net_output_log_prob, gt_prob, reduction="batchmean")
        # balance loss to avoid issues
        loss = loss * 0.03
        return loss * self.loss_weight
