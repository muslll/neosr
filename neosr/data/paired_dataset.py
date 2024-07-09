import random
import time
from typing import Any

from torch import Tensor
from torch.utils import data
from torchvision.transforms.functional import normalize

from neosr.data.data_util import (
    paired_paths_from_folder,
    paired_paths_from_lmdb,
    paired_paths_from_meta_info_file,
)
from neosr.data.file_client import FileClient
from neosr.data.transforms import basic_augment, paired_random_crop
from neosr.utils import get_root_logger, imfrombytes, img2tensor
from neosr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class paired(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
    ----
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        patch_size (int): Cropped patched size for lq patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.

    """

    def __init__(self, opt: dict[Any, Any]) -> None:
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt: dict[str, str] | None = opt.get("io_backend")
        # default to 'disk' if not specified
        if self.io_backend_opt is None:
            self.io_backend_opt = {"type": "disk"}
        # mean and std for normalizing the input images
        self.mean = opt.get("mean")
        self.std = opt.get("std")
        self.color = self.opt.get("color", None) != "y"
        self.gt_folder, self.lq_folder = opt["dataroot_gt"], opt["dataroot_lq"]

        # file client (lmdb io backend)
        if self.io_backend_opt["type"] == "lmdb":
            self.io_backend_opt["db_paths"] = [self.lq_folder, self.gt_folder]  # type: ignore[assignment]
            self.io_backend_opt["client_keys"] = ["lq", "gt"]  # type: ignore[assignment]
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ["lq", "gt"]
            )
        elif "meta_info" in self.opt and self.opt.get("meta_info", None) is not None:
            # disk backend with meta_info
            # Each line in the meta_info describes the relative path to an image
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder],
                ["lq", "gt"],
                self.opt["meta_info_file"],
            )

        else:
            # disk backend
            # it will scan the whole folder to get meta info
            # it will be time-consuming for folders with too many files. It is recommended using an extra meta txt file
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ["lq", "gt"]
            )

    def __getitem__(self, index: int) -> dict[str, str | Tensor]:
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"),  # type: ignore[union-attr]
                **self.io_backend_opt,
            )

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]["gt_path"]
        img_bytes = self.file_client.get(gt_path, "gt")  # type: ignore[attr-defined]

        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except AttributeError:
            raise AttributeError(gt_path)

        lq_path = self.paths[index]["lq_path"]
        img_bytes = self.file_client.get(lq_path, "lq")  # type: ignore[attr-defined]

        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except AttributeError:
            raise AttributeError(lq_path)

        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                if img_bytes is None:
                    msg = f"No data returned from path: {gt_path}, {lq_path}"
                    raise ValueError(msg)
            except OSError as e:
                logger = get_root_logger()
                logger.warning(
                    f"File client error: {e} in paths {gt_path}, {lq_path}, remaining retry times: {retry - 1}"
                )
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = gt_path[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1

        scale = self.opt["scale"]
        # augmentation for training
        if self.opt["phase"] == "train":
            patch_size = self.opt["patch_size"]
            flip = self.opt.get("use_hflip", True)
            rot = self.opt.get("use_rot", True)

            # random crop
            img_gt, img_lq = paired_random_crop(
                img_gt, img_lq, patch_size, scale, gt_path
            )
            # flip, rotation
            img_gt, img_lq = basic_augment([img_gt, img_lq], hflip=flip, rotation=rot)

        # crop the unmatched GT images during validation or testing
        if self.opt["phase"] != "train":
            img_gt = img_gt[0 : img_lq.shape[0] * scale, 0 : img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor(
            [img_gt, img_lq], bgr2rgb=True, float32=True, color=self.color
        )

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {"lq": img_lq, "gt": img_gt, "lq_path": lq_path, "gt_path": gt_path}

    def __len__(self) -> int:
        return len(self.paths)
