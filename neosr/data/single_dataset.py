from pathlib import Path
from typing import Any

from torch.utils import data
from torchvision.transforms.functional import normalize

from neosr.data.data_util import paths_from_lmdb
from neosr.data.file_client import FileClient
from neosr.utils import imfrombytes, img2tensor, scandir
from neosr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class single(data.Dataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
    ----
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.

    """

    def __init__(self, opt: dict[Any, Any]) -> None:
        super().__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt: dict[str, str] | None = opt.get("io_backend")
        # default to 'disk' if not specified
        if self.io_backend_opt is None:
            self.io_backend_opt = {"type": "disk"}
        self.mean = opt.get("mean")
        self.std = opt.get("std")
        self.lq_folder = opt["dataroot_lq"]
        self.color = self.opt.get("color", None) != "y"

        if self.io_backend_opt["type"] == "lmdb":
            self.io_backend_opt["db_paths"] = [self.lq_folder]  # type: ignore[assignment]
            self.io_backend_opt["client_keys"] = ["lq"]  # type: ignore[assignment]
            self.paths = paths_from_lmdb(self.lq_folder)
        elif "meta_info_file" in self.opt:
            with Path.open(self.opt["meta_info_file"], encoding="locale") as fin:
                self.paths = [
                    Path(self.lq_folder) / line.rstrip().split(" ")[0] for line in fin
                ]
        else:
            self.paths = sorted(scandir(self.lq_folder, full_path=True))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"),  # type: ignore[union-attr]
                **self.io_backend_opt,
            )

        # load lq image
        lq_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path, "lq")  # type: ignore[attr-defined]

        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except AttributeError:
            raise AttributeError(lq_path)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True, color=self.color)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
        return {"lq": img_lq, "lq_path": lq_path}

    def __len__(self) -> int:
        return len(self.paths)
