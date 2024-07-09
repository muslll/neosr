import importlib
from copy import deepcopy
from pathlib import Path
from typing import Any

from neosr.utils import get_root_logger, scandir
from neosr.utils.registry import LOSS_REGISTRY

__all__ = ["build_loss"]

# automatically scan and import loss modules for registry
# scan all the files under the 'losses' folder and collect files ending with '_loss.py'
loss_folder = Path(Path(__file__).resolve()).parent
loss_filenames = [
    Path(Path(v).name).stem for v in scandir(str(loss_folder)) if v.endswith("_loss.py")
]
# import all the loss modules
_model_modules = [
    importlib.import_module(f"neosr.losses.{file_name}") for file_name in loss_filenames
]


def build_loss(opt: dict[str, Any]):
    """Build loss from options.

    Args:
    ----
        opt (dict): Configuration. It must contain:
            type (str): Model type.

    """
    opt = deepcopy(opt)
    loss_type = opt.pop("type")
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f"Loss [{loss.__class__.__name__}] enabled.")
    return loss
