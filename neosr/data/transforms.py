import random
import cv2
import torch
import numpy as np
import torch.nn.functional as F


def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}). '
                         f'Please fix {gt_path}.')

    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size,
                     left:left + lq_patch_size] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left +
                     lq_patch_size, ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size,
                     left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size,
                     left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


def basic_augment(imgs, hflip=True, rotation=True, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if return_status:
        return imgs, (hflip, vflip, rot90)
    else:
        return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img


# Code from CutBlur:
# https://github.com/clovaai/cutblur

def augmentations(
    im1, im2,
    augs, probs, alphas,
    aux_prob=None, aux_alpha=None,
    mix_p=None
):
    idx = np.random.choice(len(augs), p=mix_p)
    aug = augs[idx]
    prob = float(probs[idx])
    alpha = float(alphas[idx])

    if aug == "none":
        im1_aug, im2_aug = im1.clone(), im2.clone()
    elif aug == "blend":
        im1_aug, im2_aug = blend(
            im1.clone(), im2.clone(),
            prob=prob, alpha=alpha
        )
    elif aug == "mixup":
        im1_aug, im2_aug, = mixup(
            im1.clone(), im2.clone(),
            prob=prob, alpha=alpha,
        )
    elif aug == "cutmix":
        im1_aug, im2_aug = cutmix(
            im1.clone(), im2.clone(),
            prob=prob, alpha=alpha,
        )
    elif aug == "cutmixup":
        im1_aug, im2_aug = cutmixup(
            im1.clone(), im2.clone(),
            mixup_prob=aux_prob, mixup_alpha=aux_alpha,
            cutmix_prob=prob, cutmix_alpha=alpha,
        )
    elif aug == "cutblur":
        im1_aug, im2_aug = cutblur(
            im1.clone(), im2.clone(),
            prob=prob, alpha=alpha
        )
    elif aug == "rgb":
        im1_aug, im2_aug = rgb(
            im1.clone(), im2.clone(),
            prob=prob
        )
    else:
        raise ValueError("{} is not invalid.".format(aug))

    return im1_aug, im2_aug, aug


def blend(im1, im2, prob=1.0, alpha=0.6):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    c = torch.empty((im2.size(0), 3, 1, 1), device=im2.device).uniform_(0, 255)
    rim2 = c.repeat((1, 1, im2.size(2), im2.size(3)))
    rim1 = c.repeat((1, 1, im1.size(2), im1.size(3)))

    v = np.random.uniform(alpha, 1)
    im1 = v * im1 + (1-v) * rim1
    im2 = v * im2 + (1-v) * rim2

    return im1, im2


def mixup(im1, im2, prob=1.0, alpha=1.2):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    v = np.random.beta(alpha, alpha)
    r_index = torch.randperm(im1.size(0)).to(im2.device)

    im1 = v * im1 + (1-v) * im1[r_index, :]
    im2 = v * im2 + (1-v) * im2[r_index, :]
    return im1, im2


def _cutmix(im2, prob=1.0, alpha=1.0):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return None

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = im2.size(2), im2.size(3)
    ch, cw = int(h*cut_ratio), int(w*cut_ratio)

    fcy = np.random.randint(0, h-ch+1)
    fcx = np.random.randint(0, w-cw+1)
    tcy, tcx = fcy, fcx
    rindex = torch.randperm(im2.size(0)).to(im2.device)

    return {
        "rindex": rindex, "ch": ch, "cw": cw,
        "tcy": tcy, "tcx": tcx, "fcy": fcy, "fcx": fcx,
    }


def cutmix(im1, im2, prob=1.0, alpha=1.0):
    c = _cutmix(im2, prob, alpha)
    if c is None:
        return im1, im2

    scale = im1.size(2) // im2.size(2)
    rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch*scale, cw*scale
    hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

    im2[..., tcy:tcy+ch, tcx:tcx+cw] = im2[rindex, :, fcy:fcy+ch, fcx:fcx+cw]
    im1[..., htcy:htcy+hch, htcx:htcx+hcw] = im1[rindex, :, hfcy:hfcy+hch, hfcx:hfcx+hcw]

    return im1, im2


def cutmixup(
    im1, im2,
    mixup_prob=1.0, mixup_alpha=1.0,
    cutmix_prob=1.0, cutmix_alpha=1.0
):
    c = _cutmix(im2, cutmix_prob, cutmix_alpha)
    if c is None:
        return im1, im2

    scale = im1.size(2) // im2.size(2)
    rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch*scale, cw*scale
    hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

    v = np.random.beta(mixup_alpha, mixup_alpha)
    if mixup_alpha <= 0 or np.random.rand(1) >= mixup_prob:
        im2_aug = im2[rindex, :]
        im1_aug = im1[rindex, :]

    else:
        im2_aug = v * im2 + (1-v) * im2[rindex, :]
        im1_aug = v * im1 + (1-v) * im1[rindex, :]

    # apply mixup to inside or outside
    if np.random.random() > 0.5:
        im2[..., tcy:tcy+ch, tcx:tcx+cw] = im2_aug[..., fcy:fcy+ch, fcx:fcx+cw]
        im1[..., htcy:htcy+hch, htcx:htcx+hcw] = im1_aug[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
    else:
        im2_aug[..., tcy:tcy+ch, tcx:tcx+cw] = im2[..., fcy:fcy+ch, fcx:fcx+cw]
        im1_aug[..., htcy:htcy+hch, htcx:htcx+hcw] = im1[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
        im2, im1 = im2_aug, im1_aug

    return im1, im2


def cutblur(im1, im2, prob=1.0, alpha=1.0):
    if im1.size() != im2.size():
        raise ValueError("im1 and im2 have to be the same resolution.")

    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = im2.size(2), im2.size(3)
    ch, cw = int(h*cut_ratio), int(w*cut_ratio)
    cy = np.random.randint(0, h-ch+1)
    cx = np.random.randint(0, w-cw+1)

    # apply CutBlur to inside or outside
    if np.random.random() > 0.5:
        im2[..., cy:cy+ch, cx:cx+cw] = im1[..., cy:cy+ch, cx:cx+cw]
    else:
        im2_aug = im1.clone()
        im2_aug[..., cy:cy+ch, cx:cx+cw] = im2[..., cy:cy+ch, cx:cx+cw]
        im2 = im2_aug

    return im1, im2


def rgb(im1, im2, prob=1.0):
    if np.random.rand(1) >= prob:
        return im1, im2

    perm = np.random.permutation(3)
    im1 = im1[:, perm]
    im2 = im2[:, perm]

    return im1, im2

