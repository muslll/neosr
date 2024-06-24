import math

import torch
from torch import nn
from torch.nn import functional as F

from neosr.utils.registry import LOSS_REGISTRY


class GaussianFilter2D(nn.Module):
    def __init__(self, window_size=11, in_channels=3, sigma=1.5, padding=None):
        """2D Gaussian Filer

        Args:
            window_size (int, optional): The window size of the gaussian filter. Defaults to 11.
            in_channels (int, optional): The number of channels of the 4d tensor. Defaults to False.
            sigma (float, optional): The sigma of the gaussian filter. Defaults to 1.5.
            padding (int, optional): The padding of the gaussian filter. Defaults to None.
                If it is set to None, the filter will use window_size//2 as the padding. Another common setting is 0.
        """
        super().__init__()
        self.window_size = window_size
        if not (window_size % 2 == 1):
            raise ValueError("Window size must be odd.")
        self.padding = padding if padding is not None else window_size // 2
        self.sigma = sigma

        kernel = self._get_gaussian_window1d()
        kernel = self._get_gaussian_window2d(kernel)
        self.register_buffer(
            name="gaussian_window", tensor=kernel.repeat(in_channels, 1, 1, 1)
        )

    def _get_gaussian_window1d(self):
        sigma2 = self.sigma * self.sigma
        x = torch.arange(-(self.window_size // 2), self.window_size // 2 + 1)
        w = torch.exp(-0.5 * x**2 / sigma2)
        w = w / w.sum()
        return w.reshape(1, 1, 1, self.window_size)

    def _get_gaussian_window2d(self, gaussian_window_1d):
        w = torch.matmul(
            gaussian_window_1d.transpose(dim0=-1, dim1=-2), gaussian_window_1d
        )
        return w

    def forward(self, x):
        x = F.conv2d(
            input=x,
            weight=self.gaussian_window,
            stride=1,
            padding=self.padding,
            groups=x.shape[1],
        )
        return x


@LOSS_REGISTRY.register()
class mssim(nn.Module):
    def __init__(
        self,
        window_size=11,
        in_channels=3,
        sigma=1.5,
        K1=0.01,
        K2=0.03,
        L=1,
        padding=None,
        clip=False,
        cosim=True,
        cosim_lambda=2,
        loss_weight=1.0,
    ):
        """Adapted from 'A better pytorch-based implementation for the mean structural
            similarity. Differentiable simpler SSIM and MS-SSIM.':
                https://github.com/lartpang/mssim.pytorch

            Calculate the mean SSIM (MSSIM) between two 4D tensors.

        Args:
            window_size (int): The window size of the gaussian filter. Defaults to 11.
            in_channels (int, optional): The number of channels of the 4d tensor. Defaults to False.
            sigma (float): The sigma of the gaussian filter. Defaults to 1.5.
            K1 (float): K1 of MSSIM. Defaults to 0.01.
            K2 (float): K2 of MSSIM. Defaults to 0.03.
            L (int): The dynamic range of the pixel values (255 for 8-bit grayscale images). Defaults to 1.
            padding (int, optional): The padding of the gaussian filter. Defaults to None. If it is set to None,
                the filter will use window_size//2 as the padding. Another common setting is 0.
            clip (bool): Clips values to train range, to reduce noise.
            cosim (bool): Enables CosineSimilary on final loss, to keep better color consistency.
            cosim_lambda (float): Lambda value to increase CosineSimilarity weight.
            loss_weight (float): Weight of final loss value.
        """
        super().__init__()

        self.window_size = window_size
        self.C1 = (K1 * L) ** 2  # equ 7 in ref1
        self.C2 = (K2 * L) ** 2  # equ 7 in ref1
        self.clip = clip
        self.cosim = cosim
        self.cosim_lambda = cosim_lambda
        self.loss_weight = loss_weight

        self.gaussian_filter = GaussianFilter2D(
            window_size=window_size,
            in_channels=in_channels,
            sigma=sigma,
            padding=padding,
        )

    #@torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, x, y):
        """x, y (Tensor): tensors of shape (N,C,H,W)
        Returns: Tensor
        """
        assert x.shape == y.shape, f"x: {x.shape} and y: {y.shape} must be the same"
        assert x.ndim == y.ndim == 4, f"x: {x.ndim} and y: {y.ndim} must be 4"

        if x.type() != self.gaussian_filter.gaussian_window.type():
            x = x.type_as(self.gaussian_filter.gaussian_window)
        if y.type() != self.gaussian_filter.gaussian_window.type():
            y = y.type_as(self.gaussian_filter.gaussian_window)

        loss = 1 - self.msssim(x, y)

        return self.loss_weight * loss

    def msssim(self, x, y):
        ms_components = []
        for i, w in enumerate((0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
            ssim, cs = self._ssim(x, y)
            ssim = ssim.mean()
            cs = cs.mean()

            if i == 4:
                ms_components.append(ssim**w)
            else:
                ms_components.append(cs**w)
                padding = [s % 2 for s in x.shape[2:]]  # spatial padding
                x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=padding)
                y = F.avg_pool2d(y, kernel_size=2, stride=2, padding=padding)

        msssim = math.prod(ms_components)  # equ 7 in ref2

        # cosine similarity
        if self.cosim:
            similarity = nn.CosineSimilarity(dim=1, eps=1e-20)
            cosine_term = (1 - similarity(x, y)).mean()
            msssim = msssim - self.cosim_lambda * cosine_term

        return msssim

    def _ssim(self, x, y):

        mu_x = self.gaussian_filter(x)  # equ 14
        mu_y = self.gaussian_filter(y)  # equ 14
        sigma2_x = self.gaussian_filter(x * x) - mu_x * mu_x  # equ 15
        sigma2_y = self.gaussian_filter(y * y) - mu_y * mu_y  # equ 15
        sigma_xy = self.gaussian_filter(x * y) - mu_x * mu_y  # equ 16

        A1 = 2 * mu_x * mu_y + self.C1
        A2 = 2 * sigma_xy + self.C2
        B1 = mu_x.pow(2) + mu_y.pow(2) + self.C1
        B2 = sigma2_x + sigma2_y + self.C2

        if self.clip:
            A1 = torch.clamp(A1, 0.003921, 0.996078)

        # equ 12, 13 in ref1
        l = A1 / B1
        cs = A2 / B2
        ssim = l * cs

        return ssim, cs
