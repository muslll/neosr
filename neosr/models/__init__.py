import importlib
from copy import deepcopy
from os import path as osp

from neosr.utils import get_root_logger, scandir
from neosr.utils.registry import MODEL_REGISTRY

__all__ = ['build_model']

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0]
                   for v in scandir(model_folder)]

# import all the model modules
_model_modules = [importlib.import_module(
    f'neosr.models.{file_name}') for file_name in model_filenames]


def build_model(opt):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    opt = deepcopy(opt)
    model = MODEL_REGISTRY.get(opt['model_type'])(opt)
    logger = get_root_logger()
    logger.info(f'Using model [{model.__class__.__name__}].')
    return model
