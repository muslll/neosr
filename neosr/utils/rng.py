from collections.abc import Callable
from pathlib import Path

from numpy.random import default_rng

from neosr.utils.options import parse_options


def rng() -> Callable:
    root_path = Path(__file__).parents[2]
    opt, __ = parse_options(str(root_path), is_train=True)
    if opt is not None:
        seed: int | None = opt.get("manual_seed", None)
    return default_rng(seed=seed) if seed is not None else default_rng()  # type: ignore[return-value]
