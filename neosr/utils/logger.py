import datetime
import logging
import time
from typing import Any

from .dist_util import get_dist_info, master_only

initialized_logger: dict[Any, Any] = {}


class AvgTimer:
    def __init__(self, window: int = 200) -> None:
        self.window = window  # average window
        self.current_time: float = 0
        self.total_time: float = 0
        self.count: int = 0
        self.avg_time: float = 0
        self.start()

    def start(self) -> None:
        self.start_time = self.tic = time.time()

    def record(self) -> None:
        self.count += 1
        self.toc = time.time()
        self.current_time = self.toc - self.tic
        self.total_time += self.current_time
        # calculate average time
        self.avg_time = self.total_time / self.count

        # reset
        if self.count > self.window:
            self.count = 0
            self.total_time = 0

        self.tic = time.time()

    def get_current_time(self):
        return self.current_time

    def get_avg_time(self):
        return self.avg_time


class MessageLogger:
    """Message logger for printing.

    Args:
    ----
        opt (dict): Config. It contains the following keys:
            name (str): Exp name.
            logger (dict): Contains 'print_freq' (str) for logger interval.
            train (dict): Contains 'total_iter' (int) for total iters.
            use_tb_logger (bool): Use tensorboard logger.
        tb_logger (obj:`tb_logger`): Tensorboard logger. Default: None.
        start_iter (int): Start iter. Default: 1.

    """

    def __init__(self, opt: dict[str, Any], tb_logger, start_iter: int = 1) -> None:
        self.exp_name = opt["name"]
        self.interval = opt["logger"].get("print_freq", 100)
        self.accumulate = opt["datasets"]["train"].get("accumulate", 1)
        self.start_iter = start_iter
        self.max_iters = opt["train"]["total_iter"]
        self.use_tb_logger = opt["logger"]["use_tb_logger"]
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

    def reset_start_time(self) -> None:
        self.start_time = time.time()

    @master_only
    def __call__(self, log_vars: dict[Any, Any]):
        """Format logging message.

        Args:
        ----
            log_vars (dict): It contains the following keys:
                epoch (int): Epoch number.
                iter (int): Current iter.
                lrs (list): List for learning rates.

                time (float): Iter time.

        """
        # epoch, iter, learning rates
        epoch: int = log_vars.pop("epoch")
        current_iter: int = int(log_vars.pop("iter"))
        lrs: list[Any] = log_vars.pop("lrs")

        message = f"[ epoch:{epoch:4d} ] [ iter:{current_iter:7,d} ]"

        # time and estimated time
        if "time" in log_vars:
            iter_time = 1 / log_vars.pop("time")
            iter_time /= self.accumulate

            total_time = time.time() - self.start_time
            time_sec_avg = total_time / (
                current_iter - (self.start_iter / self.accumulate) + 1
            )
            eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            message += f" [ performance: {iter_time:.3f} it/s ] [ lr: "
            for v in lrs:
                message += f"{v:.2e}"
            message += " ] "
            message += f"[ eta: {eta_str} ] "

        # other items, especially losses
        for k, v in log_vars.items():
            message += f"[ {k}: {v:.4e} ]"
            # tensorboard logger
            if self.use_tb_logger and "debug" not in self.exp_name:
                if k.startswith("l_"):
                    self.tb_logger.add_scalar(f"losses/{k}", v, current_iter)
                else:
                    self.tb_logger.add_scalar(k, v, current_iter)
        self.logger.info(message)


@master_only
def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415

    return SummaryWriter(log_dir=log_dir)


@master_only
def init_wandb_logger(opt) -> None:
    """We now only use wandb to sync tensorboard log."""
    import wandb  # noqa: PLC0415

    logger = get_root_logger()

    project = opt["logger"]["wandb"]["project"]
    resume_id = opt["logger"]["wandb"].get("resume_id")
    if resume_id:
        wandb_id = resume_id
        resume = "allow"
        logger.warning(f"Resume wandb logger with id={wandb_id}.")
    else:
        wandb_id = wandb.util.generate_id()
        resume = "never"

    wandb.init(
        id=wandb_id,
        resume=resume,
        name=opt["name"],
        config=opt,
        project=project,
        sync_tensorboard=True,
    )

    logger.info(f"Use wandb logger with id={wandb_id}; project={project}.")


def get_root_logger(
    logger_name: str = "neosr",
    log_level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
    ----
        logger_name (str): root logger name. Default: 'neosr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
    -------
        logging.Logger: The root logger.

    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger_name in initialized_logger:
        return logger

    format_str = "%(asctime)s %(levelname)s: %(message)s"
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter(format_str, datefmt="%d-%m-%Y %I:%M %p |")
    )
    logger.addHandler(stream_handler)
    logger.propagate = False
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel("ERROR")
    elif log_file is not None:
        logger.setLevel(log_level)
        # add file handler
        file_handler = logging.FileHandler(log_file, "w")
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    initialized_logger[logger_name] = True

    return logger
