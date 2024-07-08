from torch import Tensor, nn

from neosr.losses.basic_loss import chc
from neosr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class gan_loss(nn.Module):
    """Define GAN loss.

    Args:
    ----
        gan_type (str): Support 'bce', 'mse' (l2), 'huber' and 'chc'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 0.1.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.

    """

    def __init__(
        self,
        gan_type: str = "bce",
        real_label_val: float = 1.0,
        fake_label_val: float = 0.0,
        loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss: nn.BCEWithLogitsLoss | nn.MSELoss | nn.HuberLoss | chc

        if self.gan_type == "bce":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == "mse":
            self.loss = nn.MSELoss()
        elif self.gan_type == "huber":
            self.loss = nn.HuberLoss()
        elif self.gan_type == "chc":
            self.loss = chc()
        else:
            msg = f"GAN type {self.gan_type} is not implemented."
            raise NotImplementedError(msg)

    def get_target_label(self, net_output: Tensor, target_is_real: bool) -> Tensor:
        """Get target label.

        Args:
        ----
            net_output (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
        -------
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.

        """
        target_val = self.real_label_val if target_is_real else self.fake_label_val
        return net_output.new_ones(net_output.size()) * target_val

    def forward(
        self, net_output: Tensor, target_is_real: bool, is_disc: bool = False
    ) -> Tensor:
        """Args:
        ----
            net_output (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns
        -------
            Tensor: GAN loss value.

        """
        target_label = self.get_target_label(net_output, target_is_real)
        loss = self.loss(net_output, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight
