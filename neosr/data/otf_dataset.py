import math
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils import data

from neosr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from neosr.data.file_client import FileClient
from neosr.data.transforms import basic_augment
from neosr.utils import get_root_logger, imfrombytes, img2tensor, scandir
from neosr.utils.registry import DATASET_REGISTRY
from neosr.utils.rng import rng

rng = rng()


@DATASET_REGISTRY.register()
class otf(data.Dataset):
    """OTF degradation dataset. Originally from Real-ESRGAN.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
    ----
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.

    """

    def __init__(self, opt: dict[Any, Any]) -> None:
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt: dict[str, str] | None = opt.get("io_backend")
        # default to 'disk' if not specified
        if self.io_backend_opt is None:
            self.io_backend_opt = {"type": "disk"}
        self.color = self.opt.get("color", None) != "y"
        self.gt_folder = opt["dataroot_gt"]

        if opt.get("dataroot_lq") is not None:
            msg = "'dataroot_lq' not supported by otf, please switch to paired"
            raise ValueError(msg)

        # file client (lmdb io backend)
        if self.io_backend_opt["type"] == "lmdb":
            self.io_backend_opt["db_paths"] = [self.gt_folder]  # type: ignore[assignment]
            self.io_backend_opt["client_keys"] = ["gt"]  # type: ignore[assignment]
            if not self.gt_folder.endswith(".lmdb"):
                msg = f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}"
                raise ValueError(msg)
            with Path.open(
                Path(self.gt_folder) / "meta_info.txt", encoding="locale"
            ) as fin:
                self.paths = [line.split(".")[0] for line in fin]
        elif "meta_info" in self.opt and self.opt["meta_info"] is not None:
            # disk backend with meta_info
            # Each line in the meta_info describes the relative path to an image
            with Path(self.opt["meta_info"]).open(encoding="locale") as fin:
                paths = [line.strip().split(" ")[0] for line in fin]
                self.paths = [str(Path((self.gt_folder) / v)) for v in paths]
        else:
            # disk backend
            # it will scan the whole folder to get meta info
            # it will be time-consuming for folders with too many files. It is recommended using an extra meta txt file
            self.paths = sorted(scandir(self.gt_folder, full_path=True))

        # blur settings for the first degradation
        self.blur_kernel_size = opt.get("blur_kernel_size")
        self.kernel_list = opt.get("kernel_list")
        # a list for each kernel probability
        self.kernel_prob = opt.get("kernel_prob")
        self.blur_sigma = opt.get("blur_sigma")
        # betag used in generalized Gaussian blur kernels
        self.betag_range = opt.get("betag_range")
        # betap used in plateau blur kernels
        self.betap_range = opt.get("betap_range")
        self.sinc_prob = opt.get("sinc_prob")  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt.get("blur_kernel_size2")
        self.kernel_list2 = opt.get("kernel_list2")
        self.kernel_prob2 = opt.get("kernel_prob2")
        self.blur_sigma2 = opt.get("blur_sigma2")
        self.betag_range2 = opt.get("betag_range2")
        self.betap_range2 = opt.get("betap_range2")
        self.sinc_prob2 = opt.get("sinc_prob2")

        # a final sinc filter
        self.final_sinc_prob = opt.get("final_sinc_prob")

        # kernel size ranges from 7 to 21
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        # TODO: kernel range is now hard-coded, should be in the configure file
        # convolving with pulse tensor brings no blurry effect

        # Note: this operation must run on cpu, otherwise CUDAPrefetcher will fail
        with torch.device("cpu"):
            self.pulse_tensor = torch.zeros(21, 21, dtype=torch.float32)
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index: int) -> dict[str, str | Tensor]:
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"),  # type: ignore[union-attr]
                **self.io_backend_opt,
            )

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, "gt")  # type: ignore[attr-defined]
                if img_bytes is None:
                    msg = f"No data returned from path: {gt_path}"
                    raise ValueError(msg)
            except OSError as e:
                logger = get_root_logger()
                logger.warning(
                    f"File client error: {e} in path {gt_path}, remaining retry times: {retry - 1}"
                )
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1

        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except AttributeError:
            raise AttributeError(gt_path)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = basic_augment(
            img_gt, self.opt.get("use_hflip", True), self.opt.get("use_rot", True)
        )

        # crop or pad to 512
        # TODO: 512 is hard-coded. You may change it accordingly
        h, w = img_gt.shape[0:2]
        crop_pad_size = 512
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(
                img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
            )
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top : top + crop_pad_size, left : left + crop_pad_size, ...]

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if rng.uniform() < self.opt.get("sinc_prob", None):
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = rng.uniform(np.pi / 3, np.pi)
            else:
                omega_c = rng.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None,
            )
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if rng.uniform() < self.opt.get("sinc_prob2", None):
            if kernel_size < 13:
                omega_c = rng.uniform(np.pi / 3, np.pi)
            else:
                omega_c = rng.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2,
                [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None,
            )

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if rng.uniform() < self.opt.get("final_sinc_prob", None):
            kernel_size = random.choice(self.kernel_range)
            omega_c = rng.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True, color=self.color)[0]
        # NOTE: using torch.tensor(device='cuda') won't work.
        # Keeping old constructor for now.
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return {
            "gt": img_gt,
            "kernel1": kernel,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
            "gt_path": gt_path,
        }

    def __len__(self) -> int:
        return len(self.paths)
