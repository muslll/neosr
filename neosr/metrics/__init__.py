from copy import deepcopy

from neosr.utils.registry import METRIC_REGISTRY
from .calculate import calculate_psnr, calculate_ssim, calculate_dists

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_dists']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
