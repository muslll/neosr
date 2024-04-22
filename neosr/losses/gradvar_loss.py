import torch
from torch import nn
from torch.nn import functional as F

from neosr.losses.basic_loss import chc
from neosr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class gradvarloss(nn.Module):
    """Gradient Variance Loss:
       https://arxiv.org/abs/2309.15563v2

    Args:
        patch_size (int): size of the patches extracted from the gt and predicted images
    """

    def __init__(self, patch_size=20, criterion="chc", pad=2, loss_weight=0.5):
        super(gradvarloss, self).__init__()
        self.patch_size = patch_size
        self.loss_weight = loss_weight
        self.criterion_type = criterion
        self.pad = pad

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

        # Sobel kernel for the gradient map calculation
        self.kernel_x = (
            torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.kernel_y = (
            torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        # send to cuda
        self.kernel_x = self.kernel_x.cuda()
        self.kernel_y = self.kernel_y.cuda()
        # operation for unfolding image into non overlapping patches
        self.unfold = torch.nn.Unfold(
            kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size
        )

    def forward(self, output, target):
        # converting RGB image to grayscale
        gray_output = (
            0.2989 * output[:, 0:1, :, :]
            + 0.5870 * output[:, 1:2, :, :]
            + 0.1140 * output[:, 2:, :, :]
        )
        gray_target = (
            0.2989 * target[:, 0:1, :, :]
            + 0.5870 * target[:, 1:2, :, :]
            + 0.1140 * target[:, 2:, :, :]
        )

        # calculation of the gradient maps of x and y directions
        gx_target = F.conv2d(gray_target, self.kernel_x, stride=1, padding=self.pad)
        gy_target = F.conv2d(gray_target, self.kernel_y, stride=1, padding=self.pad)
        gx_output = F.conv2d(gray_output, self.kernel_x, stride=1, padding=self.pad)
        gy_output = F.conv2d(gray_output, self.kernel_y, stride=1, padding=self.pad)

        # unfolding image to patches
        gx_target_patches = self.unfold(gx_target)
        gy_target_patches = self.unfold(gy_target)
        gx_output_patches = self.unfold(gx_output)
        gy_output_patches = self.unfold(gy_output)

        # calculation of variance of each patch
        var_target_x = torch.var(gx_target_patches, dim=1)
        var_output_x = torch.var(gx_output_patches, dim=1)
        var_target_y = torch.var(gy_target_patches, dim=1)
        var_output_y = torch.var(gy_output_patches, dim=1)

        # loss function as a "criterion" between variances of patches extracted from gradient maps
        gradvar_loss = self.criterion(var_target_x, var_output_x) + self.criterion(
            var_target_y, var_output_y
        )

        return gradvar_loss * self.loss_weight
