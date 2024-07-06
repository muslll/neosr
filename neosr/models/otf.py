import random

import torch
from torch.nn import functional as F

from neosr.data.augmentations import apply_augment
from neosr.data.degradations import (
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
)
from neosr.data.transforms import paired_random_crop
from neosr.models.image import image
from neosr.utils.diffjpeg import DiffJPEG, filter2D
from neosr.utils.registry import MODEL_REGISTRY
from neosr.utils.rng import rng

rng = rng()


@MODEL_REGISTRY.register()
class otf(image):
    """On The Fly degradations, based on RealESRGAN pipeline."""

    def __init__(self, opt) -> None:
        super().__init__(opt)
        # simulate JPEG compression artifacts
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        queue = opt["datasets"]["train"].get("queue_size", 180)
        batch = opt["datasets"]["train"]["batch_size"]
        self.queue_size = (queue // batch) * batch
        self.patch_size = opt["datasets"]["train"].get("patch_size")
        self.device = torch.device("cuda")

    @torch.no_grad()
    def _dequeue_and_enqueue(self) -> None:
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, "queue_lr"):
            assert (
                self.queue_size % b == 0
            ), f"queue size {self.queue_size} should be divisible by batch size {b}"
            self.queue_lr = torch.zeros(
                self.queue_size, c, h, w, device=self.device
            ).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(
                self.queue_size, c, h, w, device=self.device
            ).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size, device=self.device)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr : self.queue_ptr + b, :, :, :] = (
                self.lq.clone()
            )
            self.queue_gt[self.queue_ptr : self.queue_ptr + b, :, :, :] = (
                self.gt.clone()
            )
            self.queue_ptr += b

    @torch.no_grad()
    def feed_data(self, data) -> None:
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images."""
        if self.is_train:
            # training data synthesis
            self.gt = data["gt"].to(device=self.device, non_blocking=True)

            self.kernel1 = data["kernel1"].to(device=self.device, non_blocking=True)
            self.kernel2 = data["kernel2"].to(device=self.device, non_blocking=True)
            self.sinc_kernel = data["sinc_kernel"].to(
                device=self.device, non_blocking=True
            )

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt, self.kernel1)
            # random resize
            updown_type = random.choices(
                ["up", "down", "keep"],
                self.opt["datasets"]["train"].get("resize_prob", None),
            )[0]
            if updown_type == "up":
                scale = rng.uniform(
                    1, self.opt["datasets"]["train"].get("resize_range", None)[1]
                )
            elif updown_type == "down":
                scale = rng.uniform(
                    self.opt["datasets"]["train"].get("resize_range", None)[0], 1
                )
            else:
                scale = 1
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.opt["datasets"]["train"].get("gray_noise_prob", None)
            if rng.uniform() < self.opt["datasets"]["train"].get(
                "gaussian_noise_prob", None
            ):
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.opt["datasets"]["train"].get("noise_range", None),
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt["datasets"]["train"].get(
                        "poisson_scale_range", None
                    ),
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False,
                )
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(
                *self.opt["datasets"]["train"].get("jpeg_range", None)
            )
            # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if rng.uniform() < self.opt["datasets"]["train"].get(
                "second_blur_prob", None
            ):
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(
                ["up", "down", "keep"],
                self.opt["datasets"]["train"].get("resize_prob2", None),
            )[0]
            if updown_type == "up":
                scale = rng.uniform(
                    1, self.opt["datasets"]["train"].get("resize_range2", None)[1]
                )
            elif updown_type == "down":
                scale = rng.uniform(
                    self.opt["datasets"]["train"].get("resize_range2", None)[0], 1
                )
            else:
                scale = 1
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(
                out,
                size=(
                    int(ori_h / self.opt["scale"] * scale),
                    int(ori_w / self.opt["scale"] * scale),
                ),
                mode=mode,
            )
            # add noise
            gray_noise_prob = self.opt["datasets"]["train"].get(
                "gray_noise_prob2", None
            )
            if rng.uniform() < self.opt["datasets"]["train"].get(
                "gaussian_noise_prob2", None
            ):
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.opt["datasets"]["train"].get("noise_range2", None),
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt["datasets"]["train"].get(
                        "poisson_scale_range2", None
                    ),
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False,
                )

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if rng.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(["area", "bilinear", "bicubic"])
                out = F.interpolate(
                    out,
                    size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]),
                    mode=mode,
                )
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(
                    *self.opt["datasets"]["train"].get("jpeg_range2", None)
                )
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(
                    *self.opt["datasets"]["train"].get("jpeg_range2", None)
                )
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(["area", "bilinear", "bicubic"])
                out = F.interpolate(
                    out,
                    size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]),
                    mode=mode,
                )
                out = filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

            # random crop
            patch_size = self.opt["datasets"]["train"].get("patch_size")
            (self.gt), self.lq = paired_random_crop(
                [self.gt], self.lq, patch_size, self.opt["scale"]
            )

            # training pair pool
            self._dequeue_and_enqueue()
            # for the warning: grad and param do not obey the gradient layout contract
            self.lq = self.lq.contiguous()

            # augmentation error handling
            if self.aug is not None and self.patch_size % 4 != 0:
                msg = "The patch_size value must be a multiple of 4. Please change it."
                raise ValueError(msg)
            # apply augmentation
            if self.aug is not None:
                self.gt, self.lq = apply_augment(
                    self.gt,
                    self.lq,
                    scale=self.scale,
                    augs=self.aug,
                    prob=self.aug_prob,
                )
        else:
            # for paired training or validation
            self.lq = data["lq"].to(device=self.device, non_blocking=True)
            if "gt" in data:
                self.gt = data["gt"].to(device=self.device, non_blocking=True)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img) -> None:
        # do not use the synthetic process during validation
        self.is_train = False
        super().nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True
