import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class adan_sf(Optimizer):
    """Unofficial adaptation of Schedule-Free to the Adan optimizer:
        https://arxiv.org/abs/2405.15682
        https://arxiv.org/abs/2208.06677.

    This optimizer requires that .train() and .eval() be called before the
    beginning of training and evaluation respectively. The optimizer should
    also be placed in eval mode when saving checkpoints.

    Arguments:
    ---------
        params (iterable): iterable of parameters to optimize or
            dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 1.6e-3)
        betas (Tuple[float, float, float], optional): coefficients used for
            first- and second-order moments. (default: (0.98, 0.92, 0.9955))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): decoupled weight decay
            (L2 penalty) (default: 0.02)
        max_grad_norm (float, optional): value used to clip
            global grad norm (default: 0.0 no clip)
        warmup_steps (int): Enables a linear learning rate warmup (default: 0).
        r (float): Use polynomial weighting in the average
            with power r (default: 0).
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default: 2.0).
        schedule_free (bool): Whether to enable Schedule-Free (default: True)

    """

    def __init__(  # type: ignore[no-untyped-def]
        self,
        params: Iterable[Tensor],
        lr: float = 1.6e-3,
        betas: tuple[float, float, float] = (0.98, 0.92, 0.987),
        eps: float = 1e-8,
        weight_decay: float = 0.02,
        max_grad_norm: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        schedule_free: bool = True,
        **kwargs,  # noqa: ARG002
    ) -> None:
        if not max_grad_norm >= 0.0:
            msg = f"Invalid Max grad norm: {max_grad_norm}"
            raise ValueError(msg)
        if not lr >= 0.0:
            msg = f"Invalid learning rate: {lr}"
            raise ValueError(msg)
        if not eps >= 0.0:
            msg = f"Invalid epsilon value: {eps}"
            raise ValueError(msg)
        if not 0.0 <= betas[0] < 1.0:
            msg = f"Invalid beta parameter at index 0: {betas[0]}"
            raise ValueError(msg)
        if not 0.0 <= betas[1] < 1.0:
            msg = f"Invalid beta parameter at index 1: {betas[1]}"
            raise ValueError(msg)
        if not 0.0 <= betas[2] < 1.0:
            msg = f"Invalid beta parameter at index 2: {betas[2]}"
            raise ValueError(msg)

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "r": r,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
            "warmup_steps": warmup_steps,
            "train_mode": True,
            "weight_sum": 0.0,
            "lr_max": -1.0,
            "weight_lr_power": weight_lr_power,
            "schedule_free": schedule_free,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state: dict[str, bool]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("schedule_free", True)

    @torch.no_grad()
    def restart_opt(self) -> None:
        for group in self.param_groups:
            group["step"] = 0
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    # State initialization

                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    # Exponential moving average of gradient difference
                    state["exp_avg_diff"] = torch.zeros_like(p)

    @torch.no_grad()
    def eval(self) -> None:
        for group in self.param_groups:
            train_mode = group["train_mode"]
            beta1, _, _ = group["betas"]
            if train_mode:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        # Set p.data to x
                        p.data.lerp_(end=state["z"], weight=1 - 1 / beta1)
                group["train_mode"] = False

    @torch.no_grad()
    def train(self) -> None:
        for group in self.param_groups:
            train_mode = group["train_mode"]
            beta1, _, _ = group["betas"]
            if not train_mode:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        # Set p.data to y
                        p.data.lerp_(end=state["z"], weight=1 - beta1)
                group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure: Callable[..., Any] | None = None):  # type: ignore[no-untyped-def]
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():  # type: ignore[no-untyped-call]
                loss = closure()

        if self.defaults["max_grad_norm"] > 0:
            device = self.param_groups[0]["params"][0].device
            global_grad_norm = torch.zeros(1, device=device)

            max_grad_norm = torch.tensor(self.defaults["max_grad_norm"], device=device)
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        grad = p.grad
                        global_grad_norm.add_(grad.pow(2).sum())

            global_grad_norm = torch.sqrt(global_grad_norm)

            clip_global_grad_norm = torch.clamp(
                max_grad_norm / (global_grad_norm + group["eps"]), max=1.0
            ).item()
        else:
            clip_global_grad_norm = 1.0

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avg_diffs = []
            z = []
            neg_pre_grads = []

            beta1, beta2, beta3 = group["betas"]
            if self.defaults["schedule_free"]:
                warmup_steps = group["warmup_steps"]
                weight_lr_power = group["weight_lr_power"]

            # assume same step across group now to simplify things
            # per parameter step can be easily support
            # by making it tensor, or pass list into kernel
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            bias_correction1 = 1.0 - beta1 ** group["step"]
            bias_correction2 = 1.0 - beta2 ** group["step"]
            bias_correction3 = 1.0 - beta3 ** group["step"]

            if self.defaults["schedule_free"]:
                # schedule-free
                r = group["r"]
                if group["step"] < warmup_steps:
                    sched = group["step"] / warmup_steps
                else:
                    sched = 1.0

                lr = group["lr"] * sched * math.sqrt(bias_correction3)
                lr_max = group["lr_max"] = max(lr, group["lr_max"])
                weight = (group["step"] ** r) * (lr_max**weight_lr_power)
                weight_sum = group["weight_sum"] = group["weight_sum"] + weight

                try:
                    ckp1 = weight / weight_sum
                except ZeroDivisionError:
                    ckp1 = 0

                if not group["train_mode"]:
                    msg = "Not in train mode!"
                    raise ValueError(msg)
            else:
                ckp1 = None

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["exp_avg_diff"] = torch.zeros_like(p)
                    state["z"] = torch.clone(p)

                if "neg_pre_grad" not in state or group["step"] == 1:
                    state["neg_pre_grad"] = p.grad.clone().mul_(-clip_global_grad_norm)

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                exp_avg_diffs.append(state["exp_avg_diff"])
                z.append(state["z"])
                neg_pre_grads.append(state["neg_pre_grad"])

            if not params_with_grad:
                continue

            kwargs = {
                "params": params_with_grad,
                "grads": grads,
                "exp_avgs": exp_avgs,
                "exp_avg_sqs": exp_avg_sqs,
                "exp_avg_diffs": exp_avg_diffs,
                "z": z,
                "neg_pre_grads": neg_pre_grads,
                "beta1": beta1,
                "beta2": beta2,
                "beta3": beta3,
                "bias_correction1": bias_correction1,
                "bias_correction2": bias_correction2,
                "bias_correction3_sqrt": math.sqrt(bias_correction3),
                "lr": group["lr"],
                "ckp1": ckp1,
                "weight_decay": group["weight_decay"],
                "eps": group["eps"],
                "schedule_free": group["schedule_free"],
                "clip_global_grad_norm": clip_global_grad_norm,
            }

            _multi_tensor_adan(**kwargs)

        return loss


def _multi_tensor_adan(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    exp_avg_diffs: list[Tensor],
    neg_pre_grads: list[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3_sqrt: float,
    lr: float,
    ckp1: float,
    z: list[Tensor],
    weight_decay: float,
    eps: float,
    schedule_free: bool,
    clip_global_grad_norm: Tensor,
) -> None:
    if len(params) == 0:
        return

    torch._foreach_mul_(grads, clip_global_grad_norm)

    # for memory saving, we use `neg_pre_grads`
    # to get some temp variable in a inplace way
    torch._foreach_add_(neg_pre_grads, grads)

    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)  # m_t

    torch._foreach_mul_(exp_avg_diffs, beta2)
    torch._foreach_add_(exp_avg_diffs, neg_pre_grads, alpha=1 - beta2)  # diff_t

    torch._foreach_mul_(neg_pre_grads, beta2)
    torch._foreach_add_(neg_pre_grads, grads)
    torch._foreach_mul_(exp_avg_sqs, beta3)
    torch._foreach_addcmul_(
        exp_avg_sqs, neg_pre_grads, neg_pre_grads, value=1 - beta3
    )  # n_t

    denom = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denom, bias_correction3_sqrt)
    torch._foreach_add_(denom, eps)

    torch._foreach_mul_(params, 1 - lr * weight_decay)

    if schedule_free:
        step_size_diff = lr * (beta2 / bias_correction2 * (1 - ckp1))
        step_size = lr * (bias_correction1 * (1 - ckp1))
        # in-place
        torch._foreach_lerp_(params, z, weight=ckp1)
        torch._foreach_addcdiv_(params, exp_avgs, denom, value=-step_size)
        torch._foreach_addcdiv_(params, exp_avg_diffs, denom, value=-step_size_diff)
        # z step
        torch._foreach_sub_(z, grads, alpha=lr)
    else:
        step_size_diff = lr * beta2 / bias_correction2
        step_size = lr / bias_correction1
        torch._foreach_addcdiv_(params, exp_avgs, denom, value=-step_size)
        torch._foreach_addcdiv_(params, exp_avg_diffs, denom, value=-step_size_diff)

    torch._foreach_zero_(neg_pre_grads)
    torch._foreach_add_(neg_pre_grads, grads, alpha=-1.0)
