import os
import random
import time
from collections.abc import Iterator
from os import path as osp
from pathlib import Path
from typing import Any

import torch

from neosr.utils.dist_util import master_only


def set_random_seed(seed: int) -> None:
    """Set random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)


def get_time_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def mkdir_and_rename(path: str) -> None:
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
    ----
        path (str): Folder path.

    """
    if Path(path).exists():
        new_name = str(path) + "_archived_" + get_time_str()
        print(f"Path already exists. Renaming it to {new_name}", flush=True)
        Path(path).rename(new_name)
    Path(path).mkdir(parents=True, exist_ok=True)


@master_only
def make_exp_dirs(opt: dict[str, Any]) -> None:
    """Make dirs for experiments."""
    path_opt = opt["path"].copy()
    if opt["is_train"]:
        mkdir_and_rename(path_opt.pop("experiments_root"))
    else:
        mkdir_and_rename(path_opt.pop("results_root"))
    for key, path in path_opt.items():
        if (
            ("strict_load" in key)
            or ("pretrain_network" in key)
            or ("resume" in key)
            or ("param_key" in key)
        ):
            continue
        Path(path).mkdir(parents=True, exist_ok=True)


def scandir(
    dir_path: str, suffix: None = None, recursive: bool = False, full_path: bool = False
) -> Iterator[Any]:
    """Scan a directory to find the interested files.

    Args:
    ----
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
    -------
        A generator for all the interested files with relative paths.

    """
    if (suffix is not None) and not isinstance(suffix, str | tuple):
        msg = '"suffix" must be a string or tuple of strings'
        raise TypeError(msg)

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith(".") and entry.is_file():
                return_path = entry.path if full_path else osp.relpath(entry.path, root)

                if suffix is None or return_path.endswith(suffix):
                    yield return_path
            elif recursive:
                yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
            else:
                continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def check_resume(opt: dict[str, Any], resume_iter: int) -> None:
    """Check resume states and pretrain_network paths.

    Args:
    ----
        opt (dict): Options.
        resume_iter (int): Resume iteration.

    """
    if opt["path"].get("resume_state", None):
        # get all the networks
        networks = [key for key in opt if key.startswith("network_")]
        flag_pretrain = False
        for network in networks:
            if opt["path"].get(f"pretrain_{network}") is not None:
                flag_pretrain = True
        if flag_pretrain:
            print("NOTICE: pretrain_network_* is ignored during resuming.")
        # set pretrained model paths
        for network in networks:
            name = f"pretrain_{network}"
            basename = network.replace("network_", "")
            if opt["path"].get("ignore_resume_networks") is None or (
                network not in opt["path"]["ignore_resume_networks"]
            ):
                opt["path"][name] = (
                    Path(opt["path"]["models"]) / f"net_{basename}_{resume_iter}.pth"
                )

        # change param_key to params in resume
        param_keys = [key for key in opt["path"] if key.startswith("param_key")]
        for param_key in param_keys:
            if opt["path"][param_key] == "params_ema":
                opt["path"][param_key] = "params"
                # print(f'Set {param_key} to params')


def sizeof_fmt(size: int, suffix: str = "B") -> str:
    """Get human readable file size.

    Args:
    ----
        size (int): File size.
        suffix (str): Suffix. Default: 'B'.

    Return:
    ------
        str: Formatted file size.

    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(size) < 1024.0:
            return f"{size:3.1f} {unit}{suffix}"
        size //= 1024
    return f"{size:3.1f} Y{suffix}"
