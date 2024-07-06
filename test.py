import logging
import sys
from os import path as osp
from pathlib import Path
from time import time

import torch

from neosr.data import build_dataloader, build_dataset
from neosr.models import build_model
from neosr.utils import get_root_logger, get_time_str, make_exp_dirs
from neosr.utils.options import parse_options


def test_pipeline(root_path) -> None:
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.set_default_device("cuda")
    torch.backends.cudnn.benchmark = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = Path(opt["path"]["log"]) / f"test_{opt["name"]}_{get_time_str()}.log"
    logger = get_root_logger(
        logger_name="neosr", log_level=logging.INFO, log_file=log_file
    )

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt["datasets"].items()):
        test_set = build_dataset(dataset_opt)
        num_gpu = opt.get("num_gpu", "auto")
        test_loader = build_dataloader(
            test_set,
            dataset_opt,
            num_gpu=num_gpu,
            dist=opt["dist"],
            sampler=None,
            seed=opt["manual_seed"],
        )
        logger.info(f"Number of test images in {dataset_opt["name"]}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    try:
        for test_loader in test_loaders:
            test_set_name = test_loader.dataset.opt["name"]
            logger.info(f"Testing {test_set_name}...")
            start_time = time()
            model.validation(
                test_loader,
                current_iter=opt["name"],
                tb_logger=None,
                save_img=opt["val"]["save_img"],
            )
            end_time = time()
            total_time = end_time - start_time
            n_img = len(test_loader.dataset)
            fps = n_img / total_time
            logger.info(f"Inference took {total_time:.2f} seconds, at {fps:.2f} fps.")
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    root_path = Path.resolve(Path(__file__) / osp.pardir)
    test_pipeline(root_path)
