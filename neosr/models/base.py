import sys
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim.optimizer import Optimizer

from neosr.optimizers import adamw_sf, adamw_win, adan, adan_sf
from neosr.utils import get_root_logger, tc
from neosr.utils.dist_util import master_only

if TYPE_CHECKING:
    from collections.abc import Callable


class base:
    """Default model."""

    def __init__(self, opt: dict[str, Any]) -> None:
        self.opt = opt
        self.device = torch.device("cuda")
        self.is_train = opt["is_train"]
        self.optimizers: list[Any] = []
        self.schedulers: list[Any] = []
        self.optimizer_g: Optimizer
        self.optimizer_d: Optimizer
        self.log_dict: dict[str, Any]
        self.n_accumulated: int
        self._print_different_keys_loading: Callable

        if self.is_train:
            # Schedule-Free Generator
            self.sf_optim_g = opt["train"]["optim_g"].get("schedule_free", False)
            # Schedule-Free Discriminator
            self.net_d = opt.get("network_d")
            if self.net_d is not None:
                self.sf_optim_d = opt["train"]["optim_d"].get("schedule_free", False)
        else:
            self.sf_optim_g = None
            self.sf_optim_d = None

    def feed_data(self, data: dict[str, str | Tensor]) -> None:
        pass

    def optimize_parameters(self, current_iter: int) -> None:
        pass

    def get_current_visuals(self) -> OrderedDict | None:
        pass

    def save(self, epoch: int, current_iter: int) -> None:
        pass

    def dist_validation(
        self, dataloader, current_iter: int, tb_logger, save_img: bool = True
    ) -> None:
        pass

    def nondist_validation(
        self, dataloader, current_iter: int, tb_logger, save_img: bool = True
    ) -> None:
        pass

    def validation(
        self, dataloader, current_iter: int, tb_logger, save_img: bool = True
    ) -> None:
        """Validation function.

        Args:
        ----
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.

        """
        if self.opt["dist"]:
            self.dist_validation(dataloader, current_iter, tb_logger, save_img)
        else:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def _initialize_best_metric_results(self, dataset_name: str) -> None:
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        self.best_metric_results: dict[Any, Any]
        if (
            hasattr(self, "best_metric_results")
            and dataset_name in self.best_metric_results
        ):
            return
        if not hasattr(self, "best_metric_results"):
            self.best_metric_results = {}

        # add a dataset record
        record = {}
        for metric, content in self.opt["val"]["metrics"].items():
            better = content.get("better", "higher")
            init_val = float("-inf") if better == "higher" else float("inf")
            record[metric] = {"better": better, "val": init_val, "iter": -1}
        self.best_metric_results[dataset_name] = record

    def _update_best_metric_result(
        self, dataset_name, metric: str, val, current_iter: int
    ) -> None:
        if self.best_metric_results[dataset_name][metric]["better"] == "higher":
            if val >= self.best_metric_results[dataset_name][metric]["val"]:
                self.best_metric_results[dataset_name][metric]["val"] = val
                self.best_metric_results[dataset_name][metric]["iter"] = current_iter
        elif val <= self.best_metric_results[dataset_name][metric]["val"]:
            self.best_metric_results[dataset_name][metric]["val"] = val
            self.best_metric_results[dataset_name][metric]["iter"] = current_iter

    def get_current_log(self) -> dict[str, Any]:
        return self.log_dict

    def model_to_device(self, net: nn.Module) -> nn.Module:
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
        ----
            net (nn.Module)

        """
        if self.opt.get("use_amp", False) is True:
            net = net.to(  # type: ignore[call-overload]
                self.device, non_blocking=True, memory_format=torch.channels_last
            )
        else:
            net = net.to(self.device, non_blocking=True)  # type: ignore[attr-defined]

        if self.opt.get("compile", False) is True:
            net = torch.compile(net)  # type: ignore[assignment]
            # see option fullgraph=True

        if self.opt["dist"]:
            find_unused_parameters = self.opt.get("find_unused_parameters", False)
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=find_unused_parameters,
            )
        elif self.opt["num_gpu"] > 1:
            net = DataParallel(net)  # type: ignore[type-var]
        return net

    def get_optimizer(self, optim_type: str, params, lr: float, **kwargs) -> Optimizer:
        if optim_type in {"Adam", "adam"}:
            optimizer: Optimizer = torch.optim.Adam(params, lr, **kwargs)  # type: ignore[reportPrivateImportUsage]
        elif optim_type in {"AdamW", "adamw"}:
            optimizer = torch.optim.AdamW(params, lr, **kwargs)  # type: ignore[reportPrivateImportUsage]
        elif optim_type in {"NAdam", "nadam"}:
            optimizer = torch.optim.NAdam(params, lr, **kwargs)  # type: ignore[reportPrivateImportUsage]
        elif optim_type in {"Adan", "adan"}:
            optimizer = adan(params, lr, **kwargs)
        elif optim_type in {"AdamW_Win", "adamw_win"}:
            optimizer = adamw_win(params, lr, **kwargs)
        elif optim_type in {"AdamW_SF", "adamw_sf"}:
            optimizer = adamw_sf(params, lr, **kwargs)
        elif optim_type in {"Adan_SF", "adan_sf"}:
            optimizer = adan_sf(params, lr, **kwargs)
        else:
            logger = get_root_logger()
            msg = f"{tc.red}Optimizer {optim_type} is not supported yet.{tc.end}"
            logger.error(msg)
            sys.exit(1)

        return cast(Optimizer, optimizer)

    def setup_schedulers(self) -> None:
        """Set up schedulers."""
        train_opt = self.opt["train"]
        has_scheduler = self.opt["train"].get("scheduler", None)
        if has_scheduler is not None:
            scheduler_type = train_opt["scheduler"].pop("type")
            if scheduler_type in {"MultiStepLR", "multisteplr"}:
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.MultiStepLR(
                            optimizer, **train_opt["scheduler"]
                        )
                    )
            elif scheduler_type in {"CosineAnnealing", "cosineannealing"}:
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, **train_opt["scheduler"]
                        )
                    )
            else:
                logger = get_root_logger()
                msg = f"{tc.red}Scheduler {scheduler_type} is not implemented yet.{tc.end}"
                logger.error(msg)
                sys.exit(1)

    def get_bare_model(self, net: nn.Module) -> nn.Module:
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, DataParallel | DistributedDataParallel):
            net = net.module
        return net

    def _set_lr(self, lr_groups_l) -> None:
        """Set learning rate for warm-up.

        Args:
        ----
            lr_groups_l (list): List for lr_groups, each for an optimizer.

        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l, strict=True):
            for param_group, lr in zip(optimizer.param_groups, lr_groups, strict=True):
                param_group["lr"] = lr

    def _get_init_lr(self) -> list[list[Any]]:
        """Get the initial lr, which is set by the scheduler."""
        init_lr_groups_l = []
        init_lr_groups_l.extend([
            [v["initial_lr"] for v in optimizer.param_groups]
            for optimizer in self.optimizers
        ])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter: int, warmup_iter: int = -1) -> None:
        """Update learning rate.

        Args:
        ----
            current_iter (int): Current iteration.
            warmup_iter (int): Warm-up iter numbers. -1 for no warm-up.
                Default: -1.

        """
        if current_iter > 0 and self.n_accumulated == 0:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            warm_up_lr_l.extend([
                [v / warmup_iter * current_iter for v in init_lr_g]
                for init_lr_g in init_lr_g_l
            ])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [param_group["lr"] for param_group in self.optimizers[0].param_groups]

    @master_only
    def print_network(self, net: nn.Module) -> None:
        """Print the str and parameter number of a network.

        Args:
        ----
            net (nn.Module)

        """
        if isinstance(net, DataParallel | DistributedDataParallel):
            net_cls_str = f"{net.__class__.__name__} - {net.module.__class__.__name__}"
        else:
            net_cls_str = f"{net.__class__.__name__}"

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(x.numel() for x in net.parameters())

        logger = get_root_logger()
        logger.info(f"Network: {net_cls_str}, with parameters: {net_params:,d}")
        logger.info(net_str)

    @master_only
    def save_network(
        self,
        net: nn.Module | list[nn.Module],
        net_label: list[nn.Module],
        current_iter: int | str,
        param_key: str | list[str] = "params",
    ) -> None:
        """Save networks.

        Args:
        ----
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.

        """
        if current_iter == -1:
            current_iter = "latest"
        save_filename = f"{net_label}_{current_iter}.pth"
        save_path = Path(self.opt["path"]["models"]) / save_filename

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(
            param_key
        ), "The lengths of net and param_key should be the same."

        save_dict = {}
        for net_, param_key_ in zip(net, param_key, strict=True):
            net__ = self.get_bare_model(net_)
            state_dict = net__.state_dict()
            new_state_dict = OrderedDict()
            for key, param in state_dict.items():
                if key.startswith("module."):  # remove unnecessary 'module.'
                    key = key[7:]  # noqa: PLW2901
                if key.startswith("n_averaged"):  # skip n_averaged from EMA
                    continue
                new_state_dict[key] = param.cpu()
            save_dict[param_key_] = new_state_dict

        if self.sf_optim_g and self.is_train:
            self.optimizer_g.eval()  # type: ignore[attr-defined]
        if self.net_d is not None and self.sf_optim_d and self.is_train:
            self.optimizer_d.eval()  # type: ignore[attr-defined]

        # avoid occasional writing errors
        retry = 3
        logger = get_root_logger()
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except OSError:
                logger.warning(
                    f"{tc.red}Save model error. Remaining retry times: {retry - 1}{tc.end}"
                )
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f"Still cannot save {save_path}.")
            msg = f"{tc.red}Cannot save {save_path}.{tc.end}"
            logger.error(msg)
            sys.exit(1)

        if self.sf_optim_g and self.is_train:
            self.optimizer_g.train()  # type: ignore[attr-defined]
        if self.net_d is not None and self.sf_optim_d and self.is_train:
            self.optimizer_d.train()  # type: ignore[attr-defined]

    def load_network(
        self, net: nn.Module, load_path: str, param_key: str, strict: bool = True
    ) -> None:
        """Load network.

        Args:
        ----
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: None.

        """
        self.param_key = param_key
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(
            load_path, map_location=torch.device("cuda"), weights_only=True
        )

        try:
            if "params-ema" in load_net:
                param_key = "params-ema"
            elif "params_ema" in load_net:
                param_key = "params_ema"
            elif "params" in load_net:
                param_key = "params"
            else:
                param_key = self.param_key
            load_net = load_net[param_key]
        except:  # noqa: S110
            pass

        if param_key:
            logger.info(
                f"Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}]."
            )
        else:
            logger.info(f"Loading {net.__class__.__name__} model from {load_path}.")

        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith("module."):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)
        torch.cuda.empty_cache()

    @master_only
    def save_training_state(self, epoch: int, current_iter: int) -> None:
        """Save training states during training, which will be used for
        resuming.

        Args:
        ----
            epoch (int): Current epoch.
            current_iter (int): Current iteration.

        """
        if current_iter != -1:
            state = {
                "epoch": epoch,
                "iter": current_iter,
                "optimizers": [],
                "schedulers": [],
            }
            for o in self.optimizers:
                state["optimizers"].append(o.state_dict())  # type: ignore[attr-defined]
            for s in self.schedulers:
                state["schedulers"].append(s.state_dict())  # type: ignore[attr-defined]
            save_filename = f"{int(current_iter)}.state"
            save_path = Path(self.opt["path"]["training_states"]) / save_filename

            if self.sf_optim_g and self.is_train:
                self.optimizer_g.eval()  # type: ignore[attr-defined]
            if self.net_d is not None and self.sf_optim_d and self.is_train:
                self.optimizer_d.eval()  # type: ignore[attr-defined]

            # avoid occasional writing errors
            retry = 3
            logger = get_root_logger()
            while retry > 0:
                try:
                    torch.save(state, save_path)
                except OSError:
                    logger.warning(
                        f"{tc.red}Save training state error. Remaining retry times: {retry - 1}{tc.end}"
                    )
                    time.sleep(1)
                else:
                    break
                finally:
                    retry -= 1
            if retry == 0:
                logger.warning(f"Still cannot save {save_path}. Just ignore it.")
                msg = f"{tc.red}Cannot save, aborting.{tc.end}"
                logger.error(msg)
                sys.exit(1)

            if self.sf_optim_g and self.is_train:
                self.optimizer_g.train()  # type: ignore[attr-defined]
            if self.net_d is not None and self.sf_optim_d and self.is_train:
                self.optimizer_d.train()  # type: ignore[attr-defined]

    def resume_training(self, resume_state: dict[Any, Any]) -> None:
        """Reload the optimizers and schedulers for resumed training.

        Args:
        ----
            resume_state (dict): Resume state.

        """
        resume_optimizers = resume_state["optimizers"]
        resume_schedulers = resume_state["schedulers"]
        assert len(resume_optimizers) == len(
            self.optimizers
        ), "Wrong lengths of optimizers"
        assert len(resume_schedulers) == len(
            self.schedulers
        ), "Wrong lengths of schedulers"
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def reduce_loss_dict(self, loss_dict: dict[Any, Any]) -> OrderedDict:
        """Reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
        ----
            loss_dict (OrderedDict): Loss dict.

        """
        with torch.inference_mode():
            if self.opt["dist"]:
                keys = []
                _losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    _losses.append(value)
                losses = torch.stack(_losses, 0)
                losses = torch.distributed.reduce(losses, dst=0)  # type: ignore[reportAttributeAccessIssue]
                print(losses)
                if self.opt["rank"] == 0:
                    losses /= self.opt["world_size"]
                loss_dict = dict(zip(keys, losses, strict=True))

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict
