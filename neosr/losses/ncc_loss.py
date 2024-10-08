import torch
from torch import Tensor, nn

from neosr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class ncc_loss(nn.Module):
    """Normalized Cross-Correlation loss.

    Args:
    ----
        loss_weight (float): weight for the loss. Default: 1.0
    """

    def __init__(self, loss_weight: float = 1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def _cc(self, net_output: Tensor, gt: Tensor):
        # reshape
        net_output_reshaped = net_output.view(net_output.shape[1], -1)
        gt_reshaped = gt.view(gt.shape[1], -1)
        # calculate mean
        mean_net_output = torch.mean(net_output_reshaped, 1).unsqueeze(1)
        mean_gt = torch.mean(gt_reshaped, 1).unsqueeze(1)
        # cross-correlation
        cc = torch.sum(
            (net_output_reshaped - mean_net_output) * (gt_reshaped - mean_gt), 1
        ) / torch.sqrt(
            torch.sum((net_output_reshaped - mean_net_output) ** 2, 1)
            * torch.sum((gt_reshaped - mean_gt) ** 2, 1)
        )
        return torch.mean(cc)

    def forward(self, net_output: Tensor, gt: Tensor):
        cc_value = self._cc(net_output, gt)
        return (1 - ((cc_value + 1) * 0.5)) * self.loss_weight
