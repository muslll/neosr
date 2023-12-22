import importlib
from copy import deepcopy
from os import path as osp

from neosr.utils import get_root_logger, scandir
from neosr.utils.registry import OPTIM_REGISTRY

__all__ = ['build_optimizer']

# automatically scan and import optimizer modules for registry
# scan all the files under the 'optimizers' folder and collect files ending with '_optim.py'
optim_folder = osp.dirname(osp.abspath(__file__))
optim_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(optim_folder) if v.endswith('_optim.py')]
# import all the optimizer modules
_optim_modules = [importlib.import_module(f'neosr.optimizers.{file_name}') for file_name in optim_filenames]


def build_optimizer(optim_type, params, lr, **kwargs):
    optim = OPTIM_REGISTRY.get(optim_type)(params, lr, **kwargs)
    logger = get_root_logger()
    logger.info(f'Optimizer [{optim.__class__.__name__}] is created.')
    return optim
