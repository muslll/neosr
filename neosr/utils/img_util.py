import math
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch
from numpy.typing import ArrayLike, DTypeLike
from torch import Tensor
from torchvision.utils import make_grid


def img2tensor(
    imgs: np.ndarray | list[np.ndarray],
    bgr2rgb: bool = True,
    float32: bool = True,
    color: bool = True,
) -> list[Tensor]:
    """Numpy array to tensor.

    Args:
    ----
        imgs (list[ndarray] | ndarray): Input images.
        color (bool): use RGB if true, transform to Grayscale
            if False. Default: True
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
    -------
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.

    """

    def _totensor(img, bgr2rgb, float32, color):
        if color:
            if img.shape[2] == 3 and bgr2rgb:
                if img.dtype == "float64":
                    img = img.astype("float32")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img.transpose(2, 0, 1))
        else:
            if img.shape[2] == 3 and bgr2rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = torch.from_numpy(img[None, ...])

        if float32:
            img = img.float()

        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32, color) for img in imgs]
    return _totensor(imgs, bgr2rgb, float32, color)


def tensor2img(
    tensor: Tensor | list[Tensor],
    rgb2bgr: bool = True,
    out_type: DTypeLike = np.uint8,
    min_max: tuple[int, int] = (0, 1),
) -> ArrayLike:
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
    ----
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
    -------
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.

    """
    if not (
        torch.is_tensor(tensor)
        or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))
    ):
        msg = f"tensor or list of tensors expected, got {type(tensor)}"
        raise TypeError(msg)

    if torch.is_tensor(tensor):
        tensors: Tensor | list[Tensor] = [cast(Tensor, tensor)]
    result = []
    for _tensor in tensors:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False
            ).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            elif rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            msg = f"Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}"
            raise TypeError(msg)
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def tensor2img_fast(
    tensor, rgb2bgr: bool = True, min_max: tuple[int, int] = (0, 1)
) -> ArrayLike:
    """This implementation is slightly faster than tensor2img.
    It now only supports torch tensor with shape (1, c, h, w).

    Args:
    ----
        tensor (Tensor): Now only support torch tensor with (1, c, h, w).
        rgb2bgr (bool): Whether to change rgb to bgr. Default: True.
        min_max (tuple[int]): min and max values for clamp.

    """
    output = tensor.squeeze(0).detach().clamp_(*min_max).permute(1, 2, 0)
    output = (output - min_max[0]) / (min_max[1] - min_max[0]) * 255
    output = output.type(torch.uint8).cpu().numpy()
    if rgb2bgr:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


def imfrombytes(
    content: bytes, flag: str = "color", float32: bool = False
) -> ArrayLike:
    """Read an image from bytes.

    Args:
    ----
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
    -------
        ndarray: Loaded image array.

    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        "color": cv2.IMREAD_COLOR,
        "grayscale": cv2.IMREAD_GRAYSCALE,
        "unchanged": cv2.IMREAD_UNCHANGED,
    }
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.0
    return img


def imwrite(
    img: np.ndarray,
    file_path: str,
    params: Sequence[int] | None = [],
    auto_mkdir: bool = True,
) -> None:
    """Write image to file.

    Args:
    ----
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
    -------
        bool: Successful or not.

    """
    if auto_mkdir:
        dir_name = Path(Path(file_path).parent).resolve()
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    try:
        cv2.imencode(Path(file_path).suffix, img, params or [])[1].tofile(file_path)
    except:
        msg = "Failed to write images."
        raise OSError(msg)


def crop_border(imgs: np.ndarray | list[np.ndarray], crop_border: int) -> ArrayLike:
    """Crop borders of images.

    Args:
    ----
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        crop_border (int): Crop border for each end of height and weight.

    Returns:
    -------
        list[ndarray]: Cropped images.

    """
    if crop_border == 0:
        return imgs
    if isinstance(imgs, list):
        return [
            v[crop_border:-crop_border, crop_border:-crop_border, ...] for v in imgs
        ]
    return imgs[crop_border:-crop_border, crop_border:-crop_border, ...]
