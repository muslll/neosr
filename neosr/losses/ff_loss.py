import torch
import torch.fft
from torch import Tensor, nn

from neosr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class ff_loss(nn.Module):
    """Focal Frequency Loss.
       From: https://github.com/EndlessSora/focal-frequency-loss.

    Args:
    ----
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False

    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        alpha: float = 1.0,
        patch_factor: int = 1,
        ave_spectrum: bool = True,
        log_matrix: bool = False,
        batch_matrix: bool = False,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x: Tensor) -> Tensor:
        # for amp dtype
        if x.dtype is not torch.float32:
            x = x.to(torch.float32)

        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert (
            h % patch_factor == 0
        ), "Patch factor should be divisible by image height and width"
        assert (
            w % patch_factor == 0
        ), "Patch factor should be divisible by image height and width"
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor

        patch_list.extend([
            x[:, :, i * patch_h : (i + 1) * patch_h, j * patch_w : (j + 1) * patch_w]
            for i in range(patch_factor)
            for j in range(patch_factor)
        ])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        freq = torch.fft.fft2(y, norm="ortho")
        return torch.stack([freq.real, freq.imag], -1)

    def loss_formulation(
        self, recon_freq: Tensor, real_freq: Tensor, matrix: Tensor | None = None
    ) -> Tensor:
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = (
                torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha
            )

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp /= matrix_tmp.max()
            else:
                matrix_tmp /= (
                    matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]  # noqa: PD011
                )

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert (
            weight_matrix.min().item() >= 0
        ), "The values of spectrum weight matrix should be in the range [0, 1]"
        assert (
            weight_matrix.max().item() <= 1
        ), "The values of spectrum weight matrix should be in the range [0, 1]"

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    # @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        matrix: Tensor | None = None,
        **kwargs,  # noqa: ARG002
    ) -> Tensor:
        """Forward function to calculate focal frequency loss.

        Args:
        ----
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).

        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight
