from copy import deepcopy
from typing import Any

from neosr.metrics.calculate import calculate_dists, calculate_psnr, calculate_ssim
from neosr.utils.registry import METRIC_REGISTRY

__all__ = ["calculate_dists", "calculate_psnr", "calculate_ssim"]


def calculate_metric(data, opt: dict[str, Any]):
    """Calculate metric from data and options.

    Args:
    ----
        opt (dict): Configuration. It must contain:
            type (str): Model type.

    """
    opt = deepcopy(opt)
    metric_type = opt.pop("type")
    return METRIC_REGISTRY.get(metric_type)(**data, **opt)  # type: ignore[operator]
