import importlib
import os
import random
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data
from torch.utils import data

from neosr.utils import get_root_logger, scandir
from neosr.utils.dist_util import get_dist_info
from neosr.utils.registry import DATASET_REGISTRY

__all__ = ["build_dataloader", "build_dataset"]

# automatically scan and import dataset modules for registry
# scan all the files under the data folder with '_dataset' in file names
data_folder = Path(Path(__file__).resolve()).parent
dataset_filenames = [
    Path(Path(v).name).stem
    for v in scandir(str(data_folder))
    if v.endswith("_dataset.py")
]

# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f"neosr.data.{file_name}")
    for file_name in dataset_filenames
]


def build_dataset(dataset_opt: dict[str, Any]):
    """Build dataset from options.

    Args:
    ----
        dataset_opt (dict): Configuration for dataset. It must contain:
            type (str): Dataset type.

    """
    dataset_opt = deepcopy(dataset_opt)
    dataset = DATASET_REGISTRY.get(dataset_opt["type"])(dataset_opt)
    logger = get_root_logger()
    logger.info(f"Dataset [{dataset.__class__.__name__}] is built.")
    return dataset


def build_dataloader(
    dataset: data.Dataset,
    dataset_opt: dict[str, Any],
    num_gpu: int = 1,
    dist: bool = False,
    sampler: None = None,
    seed: int | None = None,
) -> data.DataLoader:
    """Build dataloader.

    Args:
    ----
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size (int): Training batch size for each GPU.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None

    """
    phase = dataset_opt["phase"]
    rank, _ = get_dist_info()

    # train
    if phase == "train":
        if dataset_opt.get("num_worker_per_gpu", "auto") == "auto" or None:
            num_workers = len(os.sched_getaffinity(0))
        else:
            num_workers = dataset_opt["num_worker_per_gpu"]
        if dist:  # distributed training
            batch_size = dataset_opt["batch_size"]
        else:  # non-distributed training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt["batch_size"] * multiplier
            num_workers *= multiplier
        dataloader_args = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "sampler": sampler,
            "prefetch_factor": 8,
            "drop_last": True,
        }
        if sampler is None:
            dataloader_args["shuffle"] = True
        dataloader_args["worker_init_fn"] = (
            partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)
            if seed is not None
            else None
        )

    # val
    elif phase in {"val", "test"}:
        dataloader_args = {
            "dataset": dataset,
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 0,
        }
    else:
        msg = f"Wrong dataset phase: {phase}. Supported ones are 'train', 'val' and 'test'."
        raise ValueError(msg)

    dataloader_args["pin_memory"] = dataset_opt.get("pin_memory", True)
    dataloader_args["persistent_workers"] = dataset_opt.get("persistent_workers", False)

    return data.DataLoader(**dataloader_args)


def worker_init_fn(worker_id: int, num_workers: int, rank: int, seed: int) -> None:
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    # NOTE: set seed on old generator as a precaution, but
    # it is redundand since we use np.random.Generator
    np.random.seed(worker_seed)  # noqa: NPY002
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)
