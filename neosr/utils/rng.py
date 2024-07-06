from pathlib import Path

import numpy as np

from neosr.utils.options import parse_options


def rng():
    root_path = Path(__file__).parents[2]
    opt, __ = parse_options(root_path, is_train=True)
    seed = opt["manual_seed"]

    if seed is not None:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()

    return rng
