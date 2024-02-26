import torch
from torch.optim.optimizer import Optimizer

from neosr.utils.registry import OPTIMIZER_REGISTRY

@OPTIMIZER_REGISTRY.register()
class AdamL(Optimizer):
    """
    Implements AdamL optimizer, a variant of the Adam optimizer.

    AdamL adjusts the learning rate based on the adaptive and non-adaptive mode.
    The optimizer updates network weights based on their gradients.

    Parameters:
    - params (iterable): Iterable of parameters to optimize or dictionaries defining parameter groups.
    - lr (float, optional): Learning rate (default: 1e-3).
    - betas (Tuple[float, float], optional): Coefficients used for computing running averages (default: (0.9, 0.999)).
    - eps (float, optional): Term added to the denominator to improve numerical stability (default: 1e-8).
    - weight_decay (float, optional): Weight decay (L2 penalty) (default: 0).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamL, self).__init__(params, defaults)
        self.betas = betas
        self.beta_t = betas[0] ** 0  # Initialize beta1 to 1
        self.N_sma_max = 2 / (1 - betas[1]) - 1  # Compute max length for simple moving average (SMA)
        self.steps = 0  # Initialize step counter

    def step(self, closure=None):
        """
        Performs a single optimization step (parameter update).
        Parameters:
        - closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue  # Skip update if gradient is not available
                # Initialize or retrieve state
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['step'] += 1  # Increment step counter
                # Update biased first moment estimate
                exp_avg = state['exp_avg']
                beta1 = self.beta_t
                exp_avg.mul_(beta1).add_(p.grad, alpha=1.0 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq = state['exp_avg_sq']
                beta2 = self.betas[1]
                exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1.0 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                # Apply updates based on adaptive/non-adaptive mode
                if state['step'] > self.N_sma_max:
                    # Adaptive mode
                    step_size = group['lr'] / denom
                    scaled_update = exp_avg / denom
                    scaled_update.mul_(-step_size)
                    p.data.add_(scaled_update)
                else:
                    # Non-adaptive mode
                    step_size = group['lr']
                    p.data.add_(p.grad, alpha=-step_size)

        return loss
