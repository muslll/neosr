# Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py
import functools
import os
import subprocess  # noqa: S404
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_dist(launcher, backend: str = "nccl", **kwargs) -> None:
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    if launcher == "pytorch":
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == "slurm":
        _init_dist_slurm(backend, **kwargs)
    else:
        msg = f"Invalid launcher type: {launcher}"
        raise ValueError(msg)


def _init_dist_pytorch(backend: str, **kwargs) -> None:
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend: str, port: int) -> None:
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
    ----
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.

    """
    proc_id = int(os.environ["SLURM_PROCID"])
    ntasks = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")  # noqa: S605
    # specify master port
    if port is not None:
        os.environ["MASTER_PORT"] = str(port)
    elif "MASTER_PORT" in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ["MASTER_PORT"] = "29500"
    os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = str(ntasks)
    os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
    os.environ["RANK"] = str(proc_id)
    dist.init_process_group(backend=backend)


def get_dist_info() -> tuple[int, int]:
    initialized = dist.is_initialized() if dist.is_available() else False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)
        return None

    return wrapper
