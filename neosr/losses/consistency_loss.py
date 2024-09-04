from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torchvision.transforms import GaussianBlur

from neosr.losses.basic_loss import chc_loss
from neosr.utils.registry import LOSS_REGISTRY

if TYPE_CHECKING:
    from collections.abc import Callable


@LOSS_REGISTRY.register()
class consistency_loss(nn.Module):
    """Color and Luma Consistency loss using Oklab and CIE L*.

    Args:
    ----
        criterion (str): loss type. Default: 'huber'
        avgpool (bool): apply downscaling after conversion. Default: False
        scale (int): value used by avgpool. Default: 4
        loss_weight (float): weight for colorloss. Default: 1.0

    """

    def __init__(
        self,
        criterion: str = "chc",
        blur: bool = True,
        cosim: bool = True,
        saturation: float = 1.0,
        brightness: float = 1.0,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_blur = blur
        self.cosim = cosim
        self.saturation = saturation
        self.brightness = brightness
        self.loss_weight = loss_weight
        self.mean = torch.tensor((0.5, 0.5)).view(1, 2, 1, 1)
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-20)

        if self.use_blur:
            self.blur = GaussianBlur(21, 3)

        self.criterion_type = criterion
        self.criterion: nn.L1Loss | nn.HuberLoss | Callable

        if self.criterion_type == "l1":
            self.criterion = nn.L1Loss()
        elif self.criterion_type == "chc":
            self.criterion = chc_loss(loss_lambda=0, clip_min=0, clip_max=1)  # type: ignore[reportCallIssue]
        else:
            msg = f"{criterion} criterion has not been supported."
            raise NotImplementedError(msg)

    def lin_rgb(self, img: Tensor) -> Tensor:
        """Transforms sRGB gamma 2.4 to linear

        Args:
            img: Tensor (B,C,H,W).
        Returns:
            Tensor (B,C,H,W).

        """
        return torch.where(
            img <= 0.04045, img / 12.92, torch.pow((img + 0.055) / 1.055, 2.4)
        )

    def rgb_to_oklab_chroma(self, img: Tensor) -> Tensor:
        """RGB to Oklab chroma

        Args:
            img: Tensor (B,3,H,W).
        Returns:
            Tensor (B,2,H,W).

        """
        if not isinstance(img, torch.Tensor):
            msg = f"Input type is not a Tensor. Got {type(img)}"
            raise TypeError(msg)
        if len(img.shape) < 3 or img.shape[-3] != 3:
            msg = f"Input size must have a shape of (*, 3, H, W). Got {img.shape}"
            raise ValueError(msg)

        # linearize rgb
        img = self.lin_rgb(img)

        # separate into R, G, B
        r = img[:, 0, :, :]
        g = img[:, 1, :, :]
        b = img[:, 2, :, :]

        # to oklab
        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

        l_ = l.sign() * l.abs().pow(1 / 3)
        m_ = m.sign() * m.abs().pow(1 / 3)
        s_ = s.sign() * s.abs().pow(1 / 3)

        l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

        # stacked chroma
        return torch.stack([a, b], dim=1)

    def rgb_to_l_star(self, img: Tensor) -> Tensor:
        """RGB to CIELAB L*

        Args:
            img: Tensor (B,C,H,W).
        Returns:
            Tensor (B,H,W)

        """
        if not isinstance(img, torch.Tensor):
            msg = f"Input type is not a Tensor. Got {type(img)}"
            raise TypeError(msg)

        # permute
        img = img.permute(0, 2, 3, 1)

        # linearize rgb
        img = self.lin_rgb(img)

        # convert to luma - Y axis of sRGB > XYZ standard
        img = img @ torch.tensor([0.2126, 0.7152, 0.0722])

        # convert Y to L* (from CIELAB L*a*b*)
        # NOTE: will convert from range [0, 1] to range [0, 100]
        img = torch.where(
            img <= (216 / 24389),
            img * (img * (24389 / 27)),
            # torch workaround for cube-root in negative numbers
            img.sign() * img.abs().pow(1 / 3) * 116 - 16,
        )

        # normalize to [0, 1] range again and clamp
        return torch.clamp(img / 100, 0, 1)

    def forward(self, net_output: Tensor, gt: Tensor) -> Tensor:
        """
        Args:
            net_output: Tensor. Generator output.
            gt: Tensor. Generator output.
        Returns:
            float.
        """

        # clamp
        net_output = torch.clamp(net_output, 1 / 255, 1)
        gt = torch.clamp(gt, 1 / 255, 1)

        # luma
        if self.use_blur:
            net_output_blur = torch.clamp(self.blur(net_output), 0, 1)
            gt_blur = torch.clamp(self.blur(gt), 0, 1)
            input_luma = self.rgb_to_l_star(net_output_blur)
            target_luma = self.rgb_to_l_star(gt_blur) * self.brightness
        else:
            input_luma = self.rgb_to_l_star(net_output)
            target_luma = self.rgb_to_l_star(gt) * self.brightness

        # chroma
        input_chroma = self.rgb_to_oklab_chroma(net_output)
        target_chroma = self.rgb_to_oklab_chroma(gt) * self.saturation

        # clip and normalize
        input_chroma = torch.clamp((input_chroma + self.mean * 1), 0, 1)
        target_chroma = torch.clamp((target_chroma + self.mean * 1), 0, 1)

        # loss formulation
        loss = self.criterion(input_luma, target_luma) + self.criterion(
            input_chroma, target_chroma
        )

        if self.cosim:
            # cosine-similarity
            cosim_chroma = 1 - self.similarity(input_chroma, target_chroma).mean()
            cosim_luma = 1 - self.similarity(input_luma, target_luma).mean()
            # hardcoded lambda for now, as values above 0.5 causes instability
            cosim = (0.5 * cosim_chroma) + (0.5 * cosim_luma)
            # set threshold to avoid instability on early iters
            if cosim < 1e-3:
                loss = loss + cosim

        return loss * self.loss_weight
