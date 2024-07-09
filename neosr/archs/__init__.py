import importlib
from copy import deepcopy
from pathlib import Path
from typing import Any

from neosr.utils import get_root_logger, scandir
from neosr.utils.registry import ARCH_REGISTRY

__all__ = ["build_network"]


def build_network(opt: dict[str, Any]):
    # automatically scan and import arch modules for registry
    # scan all the files under the 'archs' folder and collect files ending with '_arch.py'
    arch_folder = Path(Path(__file__).resolve()).parent
    arch_filenames = [
        Path(Path(v).name).stem
        for v in scandir(str(arch_folder))
        if v.endswith("_arch.py")
    ]
    # import all the arch modules
    _arch_modules = [
        importlib.import_module(f"neosr.archs.{file_name}")
        for file_name in arch_filenames
    ]
    if opt is not None:
        opt = deepcopy(opt)
    network_type = opt.pop("type")
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f"Using network [{net.__class__.__name__}].")
    return net
