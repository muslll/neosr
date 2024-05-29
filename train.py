import datetime
import logging
import math
import sys
import time
import os
from os import path as osp

import torch

from neosr.data import build_dataloader, build_dataset
from neosr.data.data_sampler import EnlargedSampler
from neosr.data.prefetch_dataloader import CUDAPrefetcher
from neosr.models import build_model
from neosr.utils import (
    AvgTimer,
    MessageLogger,
    check_resume,
    get_root_logger,
    get_time_str,
    init_tb_logger,
    init_wandb_logger,
    make_exp_dirs,
    mkdir_and_rename,
    scandir,
)
from neosr.utils.options import copy_opt_file, parse_options


def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (
        (opt["logger"].get("wandb") is not None)
        and (opt["logger"]["wandb"].get("project") is not None)
        and ("debug" not in opt["name"])
    ):
        assert (
            opt["logger"].get("use_tb_logger") is True
        ), "should turn on tensorboard when using wandb"
        init_wandb_logger(opt)
    tb_logger = None
    if opt["logger"].get("use_tb_logger") and "debug" not in opt["name"]:
        tb_logger = init_tb_logger(
            log_dir=osp.join(opt["root_path"], "experiments", "tb_logger", opt["name"])
        )
    return tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            dataset_enlarge_ratio = dataset_opt.get("dataset_enlarge_ratio", 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(
                train_set, opt["world_size"], opt["rank"], dataset_enlarge_ratio
            )
            num_gpu = opt.get("num_gpu", "auto")
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=num_gpu,
                dist=opt["dist"],
                sampler=train_sampler,
                seed=opt["manual_seed"],
            )

            accumulate = opt["datasets"]["train"].get("accumulate", 1)
            num_iter_per_epoch = math.ceil(
                len(train_set)
                * dataset_enlarge_ratio
                / (dataset_opt["batch_size"] * accumulate * opt["world_size"])
            )
            total_iters = int(opt["train"]["total_iter"] * accumulate)
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tStarting model: {opt["name"]}'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size"]}'
                f'\n\tAccumulated batches: {dataset_opt["batch_size"] * accumulate}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequired iters per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}. Total iters: {total_iters // accumulate}'
            )
        elif phase.split("_")[0] == "val":
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt["num_gpu"],
                dist=opt["dist"],
                sampler=None,
                seed=opt["manual_seed"],
            )
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}'
            )
            val_loaders.append(val_loader)
        else:
            raise ValueError(f"Dataset phase {phase} is not recognized.")

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt["auto_resume"]:
        state_path = osp.join("experiments", opt["name"], "training_states")
        if osp.isdir(state_path):
            states = list(
                scandir(state_path, suffix="state", recursive=False, full_path=False)
            )
            if len(states) != 0:
                states = [float(v.split(".state")[0]) for v in states]
                resume_state_path = osp.join(state_path, f"{max(states):.0f}.state")
                opt["path"]["resume_state"] = resume_state_path

    else:
        if opt["path"].get("resume_state"):
            resume_state_path = opt["path"]["resume_state"]

    if resume_state_path is None:
        resume_state = None
    else:
        resume_state = torch.load(resume_state_path, map_location=torch.device("cuda"))
        check_resume(opt, resume_state["iter"])
    return resume_state


def train_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True)
    opt["root_path"] = root_path

    # default device
    torch.set_default_device("cuda")

    # enable tensorfloat32 and possibly bfloat16 matmul
    fast_matmul = opt.get("fast_matmul", False)
    if fast_matmul:
        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if (
            opt["logger"].get("use_tb_logger")
            and "debug" not in opt["name"]
            and opt["rank"] == 0
        ):
            mkdir_and_rename(
                osp.join(opt["root_path"], "experiments", "tb_logger", opt["name"])
            )

    # copy the yml file to the experiment root
    try:
        copy_opt_file(args.opt, opt["path"]["experiments_root"])
    except Exception as e:
        msg = "Failed. Make sure the option 'name' in your config file is the same as the previous state!"
        raise ValueError(msg)

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt["path"]["log"], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name="neosr", log_level=logging.INFO, log_file=log_file
    )
    logger.info(
        f"\n------------------------ neosr ------------------------\nPytorch Version: {torch.__version__}"
    )
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # create model
    model = build_model(opt)

    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(
            f"Resuming training from epoch: {resume_state['epoch']}, iter: {int(resume_state['iter'])}"
        )
        start_epoch = resume_state["epoch"]
        current_iter = int(resume_state["iter"] * opt["datasets"]["train"].get("accumulate", 1))
        #current_iter = resume_state["iter"]
        torch.cuda.empty_cache()
    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetcher = CUDAPrefetcher(train_loader, opt)

    if opt["use_amp"]:
        logger.info("AMP enabled.")

    if opt["deterministic"]:
        logger.info("Deterministic mode enabled.")

    # training log vars
    accumulate = opt["datasets"]["train"].get("accumulate", 1)
    print_freq = opt["logger"]["print_freq"]
    save_checkpoint_freq = opt["logger"]["save_checkpoint_freq"]
    if opt.get("val") is not None:
        val_freq = opt["val"]["val_freq"]

    # training
    logger.info(f"Start training from epoch: {start_epoch}, iter: {int(current_iter / accumulate)}")
    #data_timer, iter_timer = AvgTimer(), AvgTimer()
    iter_timer = AvgTimer()
    start_time = time.time()

    try:
        for epoch in range(start_epoch, total_epochs + 1):
            train_sampler.set_epoch(epoch)
            prefetcher.reset()
            train_data = prefetcher.next()

            while train_data is not None:
                #data_timer.record()

                current_iter += 1
                if current_iter > total_iters:
                    break
                # training
                model.feed_data(train_data)
                model.optimize_parameters(current_iter)
                # update learning rate
                model.update_learning_rate(
                    current_iter, warmup_iter=opt["train"].get("warmup_iter", -1)
                )
                iter_timer.record()
                if current_iter == 1:
                    # reset start time in msg_logger for more accurate eta_time
                    # not work in resume mode
                    msg_logger.reset_start_time()

                # log
                if current_iter >= accumulate:
                    current_iter_log = current_iter / accumulate
                else:
                    current_iter_log = current_iter

                if current_iter_log % print_freq == 0:
                    log_vars = {"epoch": epoch, "iter": current_iter_log}
                    log_vars.update({"lrs": model.get_current_learning_rate()})
                    log_vars.update({
                        "time": iter_timer.get_avg_time(),
                        #"data_time": data_timer.get_avg_time(),
                    })
                    log_vars.update(model.get_current_log())
                    msg_logger(log_vars)

                # save models and training states
                if current_iter_log % save_checkpoint_freq == 0:
                    logger.info("Saving models and training states.")
                    model.save(epoch, int(current_iter_log))

                # validation
                if opt.get("val") is not None and (current_iter_log % val_freq == 0):
                    for val_loader in val_loaders:
                        model.validation(
                            val_loader,
                            int(current_iter_log),
                            tb_logger,
                            opt["val"]["save_img"],
                        )

                #data_timer.start()
                iter_timer.start()
                train_data = prefetcher.next()
            # end of iter

        # end of epoch

        consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        logger.info(f"End of training. Time consumed: {consumed_time}")
        logger.info("Save the latest model.")
        model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest

    except KeyboardInterrupt:
        logger.info("Interrupted, saving latest models.")
        model.save(epoch, int(current_iter_log))
        sys.exit(0)

    if opt.get("val") is not None:
        accumulate = opt["datasets"]["train"].get("accumulate", 1)
        for val_loader in val_loaders:
            model.validation(
                val_loader, int(current_iter / accumulate), tb_logger, opt["val"]["save_img"]
            )
    if tb_logger:
        tb_logger.close()


if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)
