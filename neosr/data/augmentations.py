import random

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from neosr.utils.rng import rng

rng = rng()


@torch.no_grad()
def mixup(
    img_gt: Tensor, img_lq: Tensor, alpha_min: float = 0.4, alpha_max: float = 0.6
) -> tuple[Tensor, Tensor]:
    r"""MixUp augmentation.

    "Mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)".
    In ICLR, 2018.
        https://github.com/facebookresearch/mixup-cifar10

    Args:
    ----
        img_gt, img_lq (Tensor): Input images of shape (N, C, H, W).
            Assumes same size.
        alpha_min/max (float): The given min/max mixing ratio.

    """
    if img_gt.size() != img_lq.size():
        msg = "img_gt and img_lq have to be the same resolution."
        raise ValueError(msg)

    lam = rng.uniform(alpha_min, alpha_max)

    # mixup process
    rand_index = torch.randperm(img_gt.size(0))
    img_ = img_gt[rand_index]

    img_gt = lam * img_gt + (1 - lam) * img_
    img_lq = lam * img_lq + (1 - lam) * img_

    return img_gt, img_lq


@torch.no_grad()
def cutmix(img_gt: Tensor, img_lq: Tensor, alpha: float = 0.9) -> tuple[Tensor, Tensor]:
    r"""CutMix augmentation.

    "CutMix: Regularization Strategy to Train Strong Classifiers with
    Localizable Features (https://arxiv.org/abs/1905.04899)". In ICCV, 2019.
        https://github.com/clovaai/CutMix-PyTorch

    Args:
    ----
        img_gt, img_lq (Tensor): Input images of shape (N, C, H, W).
            Assumes same size.
        alpha (float): The given maximum mixing ratio.

    """
    if img_gt.size() != img_lq.size():
        msg = "img_gt and img_lq have to be the same resolution."
        raise ValueError(msg)

    def rand_bbox(size, lam):
        """Generate random box by lam."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = rng.integers(W)
        cy = rng.integers(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    lam = rng.uniform(0, alpha)
    rand_index = torch.randperm(img_gt.size(0))

    # mixup process
    img_gt_ = img_gt[rand_index]
    img_lq_ = img_lq[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(img_gt.size(), lam)
    img_gt[:, :, bbx1:bbx2, bby1:bby2] = img_gt_[:, :, bbx1:bbx2, bby1:bby2]
    img_lq[:, :, bbx1:bbx2, bby1:bby2] = img_lq_[:, :, bbx1:bbx2, bby1:bby2]

    return img_gt, img_lq


@torch.no_grad()
def resizemix(
    img_gt: Tensor, img_lq: Tensor, scope: tuple[float, float] = (0.2, 0.9)
) -> tuple[Tensor, Tensor]:
    r"""ResizeMix augmentation.

    "ResizeMix: Mixing Data with Preserved Object Information and True Labels
    (https://arxiv.org/abs/2012.11101)".

    Args:
    ----
        img_gt, img_lq (Tensor): Input images of shape (N, C, H, W).
            Assumes same size.
        scope (float): The given maximum mixing ratio.

    """
    if img_gt.size() != img_lq.size():
        msg = "img_gt and img_lq have to be the same resolution."
        raise ValueError(msg)

    def rand_bbox_tao(size, tao):
        """Generate random box by tao (scale)."""
        W = size[2]
        H = size[3]
        cut_w = int(W * tao)
        cut_h = int(H * tao)

        # uniform
        cx = rng.integers(W)
        cy = rng.integers(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    # index
    rand_index = torch.randperm(img_gt.size(0))
    img_gt_resize = img_gt.clone()
    img_gt_resize = img_gt_resize[rand_index]
    img_lq_resize = img_lq.clone()
    img_lq_resize = img_lq_resize[rand_index]

    # generate tao
    tao = rng.uniform(scope[0], scope[1])

    # random box
    bbx1, bby1, bbx2, bby2 = rand_bbox_tao(img_gt.size(), tao)

    # resize
    img_gt_resize = torch.clamp(
        F.interpolate(
            img_gt_resize, (bby2 - bby1, bbx2 - bbx1), mode="bicubic", antialias=True
        ),
        0,
        1,
    )
    img_lq_resize = torch.clamp(
        F.interpolate(
            img_lq_resize, (bby2 - bby1, bbx2 - bbx1), mode="bicubic", antialias=True
        ),
        0,
        1,
    )

    # mix
    img_gt[:, :, bby1:bby2, bbx1:bbx2] = img_gt_resize
    img_lq[:, :, bby1:bby2, bbx1:bbx2] = img_lq_resize

    return img_gt, img_lq


@torch.no_grad()
def cutblur(
    img_gt: Tensor, img_lq: Tensor, alpha: float = 0.7
) -> tuple[Tensor, Tensor]:
    r"""CutBlur Augmentation.

    "Rethinking Data Augmentation for Image Super-resolution:
        A Comprehensive Analysis and a New Strategy"
        (https://arxiv.org/abs/2004.00448)

    Args:
    ----
        img_gt, img_lq (Tensor): Input images of shape (N, C, H, W).
            Assumes same size.
        alpha (float): The given max mixing ratio.

    """
    if img_gt.size() != img_lq.size():
        msg = "img_gt and img_lq have to be the same resolution."
        raise ValueError(msg)

    def rand_bbox(size, lam):
        """Generate random box by lam (scale)."""
        W = size[2]
        H = size[3]
        cut_w = int(W * lam)
        cut_h = int(H * lam)

        # uniform
        cx = rng.integers(W)
        cy = rng.integers(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    lam = rng.uniform(0.2, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(img_gt.size(), lam)

    # apply cutblur
    img_lq[:, :, bbx1:bbx2, bby1:bby2] = img_gt[:, :, bbx1:bbx2, bby1:bby2]

    return img_gt, img_lq


@torch.no_grad()
def apply_augment(
    img_gt: Tensor,
    img_lq: Tensor,
    scale: int = 1,
    augs: tuple[str, str, str, str, str] = (
        "none",
        "mixup",
        "cutmix",
        "resizemix",
        "cutblur",
    ),
    prob: tuple[float, float, float, float, float] = (0.1, 0.3, 0.2, 0.7, 0.8),
    multi_prob: float = 0.3,
) -> tuple[Tensor, Tensor]:
    r"""Applies Augmentations.

    Args:
    ----
        img_gt, img_lq (Tensor): Input images of shape (N, C, H, W).
        scale (int): Scale ratio between GT and LQ.
        augs (list): List of possible augmentations to apply. Supported
            values are: "none", "mixup", "cutmix", "resizemix" and "cutblur"
        prob (list|float): List of float probabilities for each augmentation.
        multi_prob (float): Probability that to apply >1 augmentations from the list.
            Default: 0.3
    Return:
        img_gt, img_lq (Tensor)

    """
    if len(augs) != len(prob):
        msg = "Length of 'augmentation' and aug_prob don't match!"
        raise ValueError(msg)
    if img_gt.shape[0] == 1:
        msg = "Augmentations need batch >1 to work."
        raise ValueError(msg)

    # match resolutions
    modes = ["bilinear", "bicubic"]
    if scale > 1:
        img_lq = torch.clamp(
            F.interpolate(
                img_lq, scale_factor=scale, mode=random.choice(modes), antialias=True
            ),
            0,
            1,
        )

    if rng.random() < multi_prob:
        num_augs = rng.integers(2, len(augs)) if len(augs) > 2 else len(augs)
        weighted = list(zip(augs, prob, strict=False))
        aug: str | list[str]
        aug = []
        for _ in range(num_augs):
            choice = random.choices(weighted, k=1)
            aug.append(choice[0][0])
            weighted.remove(choice[0])

        if "cutmix" in aug:
            img_gt, img_lq = cutmix(img_gt, img_lq)
        if "mixup" in aug:
            img_gt, img_lq = mixup(img_gt, img_lq)
        if "resizemix" in aug:
            img_gt, img_lq = resizemix(img_gt, img_lq)
        if "cutblur" in aug:
            img_gt, img_lq = cutblur(img_gt, img_lq)

    else:
        idx = random.choices(range(len(augs)), weights=prob)[0]
        aug = augs[idx]
        if "cutmix" in aug:
            img_gt, img_lq = cutmix(img_gt, img_lq)
        elif "mixup" in aug:
            img_gt, img_lq = mixup(img_gt, img_lq)
        elif "resizemix" in aug:
            img_gt, img_lq = resizemix(img_gt, img_lq)
        elif "cutblur" in aug:
            img_gt, img_lq = cutblur(img_gt, img_lq)
        else:
            pass

    # back to original resolution
    if scale > 1:
        img_lq = torch.clamp(
            F.interpolate(
                img_lq, scale_factor=1 / scale, mode="bicubic", antialias=True
            ),
            0,
            1,
        )

    return img_gt, img_lq
