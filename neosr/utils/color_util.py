import numpy as np
import torch


def rgb2ycbcr(img, y_only=False):
    """Convert a RGB image to YCbCr image.

    This function produces the same results as Matlab's `rgb2ycbcr` function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
    else:
        out_img = np.matmul(
            img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def ycbcr2rgb(img):
    """Convert a YCbCr image to RGB image.

    This function produces the same results as Matlab's ycbcr2rgb function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `YCrCb <-> RGB`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        ndarray: The converted RGB image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img) * 255
    out_img = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                              [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]  # noqa: E126
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def ycbcr2bgr(img):
    """Convert a YCbCr image to BGR image.

    The bgr version of ycbcr2rgb.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `YCrCb <-> BGR`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        ndarray: The converted BGR image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img) * 255
    out_img = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0.00791071, -0.00153632, 0],
                              [0, -0.00318811, 0.00625893]]) * 255.0 + [-276.836, 135.576, -222.921]  # noqa: E126
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].

    Args:
        img (ndarray): The input image.

    Returns:
        (ndarray): The converted image with type of float32 and range of
            [0, 1].
    """

    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.float16:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError(
            f'The img type should be np.float32, np.float16 or np.uint8, but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is float32, float16, it converts the image to
    those types with range [0, 1].

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (type): If dst_type is uint8, it converts the image to np.uint8
            type with range [0, 255]. If dst_type is float32, it converts the
            image to np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """

    if dst_type not in (np.uint8, np.float32, np.float16):
        raise TypeError(
            f'The dst_type should be np.float32, np.float16 or np.uint8, but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)


def rgb2ycbcr_pt(img, y_only=False):
    """Convert RGB images to YCbCr images (PyTorch version).

    It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    Args:
        img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
         y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img, non_blocking=True)
        out_img = torch.matmul(img.permute(0, 2, 3, 1),
                               weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor(
            [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img, non_blocking=True)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img, non_blocking=True)
        out_img = torch.matmul(img.permute(0, 2, 3, 1),
                               weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img


def rgb_to_uv(img: torch.Tensor) -> torch.Tensor:
    '''
    RGB to YUV. Outputs tensor with only UV channels. 
    '''

    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(img.shape) < 3 or img.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    # define separate rgb channels
    r: torch.Tensor = img[..., 0, :, :]
    g: torch.Tensor = img[..., 1, :, :]
    b: torch.Tensor = img[..., 2, :, :]

    # bt.709 values
    Wr = 0.2126
    Wb = 0.0722
    Wg = 1 - Wr - Wb  # 0.7152
    Uc = 0.539
    Vc = 0.635
    delta: float = 0.5

    # convert to yuv
    y: torch.Tensor = Wr * r + Wg * g + Wb * b
    u: torch.Tensor = (b - y) * Uc + delta  # cb
    v: torch.Tensor = (r - y) * Vc + delta  # cr

    # return only uv (colors)
    out_img = torch.stack((u, v), -3)

    return out_img
    
def rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    '''
    RGB to YUV. Outputs tensor with only Y channel. 
    '''

    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(img.shape) < 3 or img.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    # define separate rgb channels
    r: torch.Tensor = img[..., 0, :, :]
    g: torch.Tensor = img[..., 1, :, :]
    b: torch.Tensor = img[..., 2, :, :]

    # bt.709 values
    Wr = 0.2126
    Wb = 0.0722
    Wg = 1 - Wr - Wb  # 0.7152
    Uc = 0.539
    Vc = 0.635
    delta: float = 0.5

    # convert to yuv
    y: torch.Tensor = Wr * r + Wg * g + Wb * b
    #u: torch.Tensor = (b - y) * Uc + delta  # cb
    #v: torch.Tensor = (r - y) * Vc + delta  # cr

    # return only y (luma)
    out_img = torch.stack((y,), -3)

    return out_img
