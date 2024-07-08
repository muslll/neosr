import cv2
import numpy as np
import torch
from torch import Tensor

from neosr.losses.dists_loss import dists_loss
from neosr.metrics.metric_util import reorder_image, to_y_channel
from neosr.utils.img_util import img2tensor
from neosr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_psnr(
    img: np.ndarray,
    img2: np.ndarray,
    crop_border: int = 4,
    input_order: str = "HWC",
    test_y_channel: bool = False,
    **kwargs,  # noqa: ARG001
) -> float:
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
    ----
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
    -------
        float: PSNR result.

    """
    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."
    if input_order not in {"HWC", "CHW"}:
        msg = f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"'
        raise ValueError(msg)
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 * 255.0 / mse)


@METRIC_REGISTRY.register()
def calculate_ssim(
    img: np.ndarray,
    img2: np.ndarray,
    crop_border: int = 4,
    input_order: str = "HWC",
    test_y_channel: bool = False,
    **kwargs,  # noqa: ARG001
) -> float:
    """Calculate SSIM (structural similarity).

    ``Paper: Image quality assessment: From error visibility to structural similarity``

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
    ----
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
    -------
        float: SSIM result.

    """
    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."
    if input_order not in {"HWC", "CHW"}:
        msg = f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"'
        raise ValueError(msg)
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssims = []
    ssims.extend([_ssim(img[..., i], img2[..., i]) for i in range(img.shape[2])])
    return np.array(ssims).mean()


def _ssim(img: np.ndarray, img2: np.ndarray) -> float:
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
    ----
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
    -------
        float: SSIM result.

    """
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    # valid mode for window size 11
    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


@METRIC_REGISTRY.register()
def calculate_dists(
    img: Tensor | np.ndarray,
    img2: Tensor | np.ndarray,
    **kwargs,  # noqa: ARG001
) -> float:
    """Calculates DISTS metric.

    Args:
    ----
        img (np.ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (np.ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
    -------
        float: SSIM result.

    """
    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."

    # to tensor
    img, img2 = img2tensor([img, img2])
    # normalize to [0, 1]
    img, img2 = img / 255, img2 / 255
    # add dim
    if isinstance(img, Tensor) and isinstance(img2, Tensor):
        img, img2 = img.unsqueeze_(0), img2.unsqueeze_(0)
        # to cuda
        device = torch.device("cuda")
        img, img2 = img.to(device), img2.to(device)

    loss = dists_loss(as_loss=False)
    return loss.forward(img, img2)
