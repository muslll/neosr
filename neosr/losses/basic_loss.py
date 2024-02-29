from collections import OrderedDict

import torch
import torch.fft
from torch import nn
from torch.nn import functional as F

from neosr.archs.vgg_arch import VGGFeatureExtractor
from neosr.losses.loss_util import weighted_loss
from neosr.utils.color_util import rgb_to_cbcr, rgb_to_luma
from neosr.utils.registry import LOSS_REGISTRY

_reduction_modes = ["none", "mean", "sum"]


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction="none")


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction="none")


@weighted_loss
def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=1.0)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(L1Loss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction
        )


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(MSELoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction
        )


@LOSS_REGISTRY.register()
class HuberLoss(nn.Module):
    """HuberLoss

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        delta (float): Specifies the threshold at which to change between
            delta-scaled L1 and L2 loss. The value must be positive. Default: 1.0
    """

    def __init__(
        self, loss_weight: float = 1.0, reduction: str = "mean", delta: float = 1.0
    ) -> None:
        super(HuberLoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.delta = delta

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, weight: float = None, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """

        return self.loss_weight * huber_loss(
            pred, target, weight, delta=self.delta, reduction=self.reduction
        )


@LOSS_REGISTRY.register()
class chc(nn.Module):
    """Clipped pseudo-Huber with Cosine Similarity Loss

       For reference on research, see:
       https://github.com/HolmesShuan/AIM2020-Real-Super-Resolution
       https://github.com/dmarnerides/hdr-expandnet

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        criterion (str): Specifies the loss to apply.
            Supported choices are 'l1' and 'huber'. Default: 'huber'.
        loss_lambda (float):  constant factor that adjusts the contribution of the cosine similarity term
        clip_min (float): threshold that sets the gradients of well-trained pixels to zeros
        clip_max (float): max clip limit, can act as a noise filter
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction: str = "mean",
        criterion: str = "huber",
        loss_lambda: float = 0.019607,
        clip_min: float = 0.003921,
        clip_max: float = 0.996078,
    ) -> None:
        super(chc, self).__init__()

        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        # Loss params
        self.loss_weight = loss_weight
        self.criterion = criterion

        # CoSim
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-20)
        self.loss_lambda = loss_lambda  # 5/255 = 0.019607

        # Clip
        self.clip_min = clip_min  # 1/255 = 0.03921
        self.clip_max = clip_max

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        cosine_term = (1 - self.similarity(pred, target)).mean()

        # absolute mean
        if self.criterion == "l1":
            loss = torch.mean(
                torch.clamp(
                    (torch.abs(pred - target) + self.loss_lambda * cosine_term),
                    self.clip_min,
                    self.clip_max,
                )
            )
        # pseudo-huber (charbonnier)
        elif self.criterion == "huber":
            loss = torch.mean(
                torch.clamp(
                    (
                        torch.sqrt((pred - target) ** 2 + 1e-12)
                        + self.loss_lambda * cosine_term
                    ),
                    self.clip_min,
                    self.clip_max,
                )
            )
        else:
            raise NotImplementedError(f"{self.criterion} not implemented.")

        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(
        self,
        layer_weights: OrderedDict,
        vgg_type: str = "vgg19",
        use_input_norm: bool = True,
        range_norm: bool = False,
        perceptual_weight: float = 1.0,
        style_weight: float = 0.0,
        criterion: str = "huber",
    ) -> None:
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm,
        )

        self.criterion_type = criterion
        if self.criterion_type == "l1":
            self.criterion = nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = nn.MSELoss()
        elif self.criterion_type == "huber":
            self.criterion = nn.HuberLoss()
        elif self.criterion_type == "chc":
            self.criterion = chc()
        elif self.criterion_type == "fro":
            self.criterion = None
        else:
            raise NotImplementedError(f"{criterion} criterion has not been supported.")

    def forward(
        self, x: torch.Tensor, gt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == "fro":
                    percep_loss += (
                        torch.norm(x_features[k] - gt_features[k], p="fro")
                        * self.layer_weights[k]
                    )
                else:
                    percep_loss += (
                        self.criterion(x_features[k], gt_features[k])
                        * self.layer_weights[k]
                    )
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == "fro":
                    style_loss += (
                        torch.norm(
                            self._gram_mat(x_features[k])
                            - self._gram_mat(gt_features[k]),
                            p="fro",
                        )
                        * self.layer_weights[k]
                    )
                else:
                    style_loss += (
                        self.criterion(
                            self._gram_mat(x_features[k]),
                            self._gram_mat(gt_features[k]),
                        )
                        * self.layer_weights[k]
                    )
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


@LOSS_REGISTRY.register()
class colorloss(nn.Module):
    """Color Consistency Loss.
    Converts images to chroma-only and compares both.

    Args:
        criterion (str): loss type. Default: 'huber'
        avgpool (bool): apply downscaling after conversion. Default: False
        scale (int): value used by avgpool. Default: 4
        loss_weight (float): weight for colorloss. Default: 1.0
    """

    def __init__(
        self,
        criterion: str = "huber",
        avgpool: bool = False,
        scale: int = 4,
        loss_weight: float = 1.0,
    ) -> None:
        super(colorloss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion_type = criterion
        self.avgpool = avgpool
        self.scale = scale

        if self.criterion_type == "l1":
            self.criterion = nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = nn.MSELoss()
        elif self.criterion_type == "huber":
            self.criterion = nn.HuberLoss()
        elif self.criterion_type == "chc":
            self.criterion = chc()
        else:
            raise NotImplementedError(f"{criterion} criterion has not been supported.")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_uv = rgb_to_cbcr(input)
        target_uv = rgb_to_cbcr(target)

        # TODO: test downscale operation
        if self.avgpool:
            input_uv = torch.nn.AvgPool2d(kernel_size=int(self.scale))(input_uv)
            target_uv = torch.nn.AvgPool2d(kernel_size=int(self.scale))(target_uv)

        return self.criterion(input_uv, target_uv) * self.loss_weight


@LOSS_REGISTRY.register()
class lumaloss(nn.Module):
    """Luminance Loss.
    Converts images to Y from CIE XYZ and then to CIE L* (from L*a*b*)

    Args:
        criterion (str): loss type. Default: 'huber'
        avgpool (bool): apply downscaling after conversion. Default: False
        scale (int): value used by avgpool. Default: 4
        loss_weight (float): weight for colorloss. Default: 1.0
    """

    def __init__(
        self,
        criterion: str = "huber",
        avgpool: bool = False,
        scale: int = 4,
        loss_weight: float = 1.0,
    ) -> None:
        super(lumaloss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion_type = criterion
        self.avgpool = avgpool
        self.scale = scale

        if self.criterion_type == "l1":
            self.criterion = nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = nn.MSELoss()
        elif self.criterion_type == "huber":
            self.criterion = nn.HuberLoss()
        elif self.criterion_type == "chc":
            self.criterion = chc()
        else:
            raise NotImplementedError(f"{criterion} criterion has not been supported.")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_luma = rgb_to_luma(input)
        target_luma = rgb_to_luma(target)

        if self.avgpool:
            input_luma = torch.nn.AvgPool2d(kernel_size=int(self.scale))(input_luma)
            target_luma = torch.nn.AvgPool2d(kernel_size=int(self.scale))(target_luma)

        return self.criterion(input_luma, target_luma) * self.loss_weight


@LOSS_REGISTRY.register()
class focalfrequencyloss(nn.Module):
    """Focal Frequency Loss.
       From: https://github.com/EndlessSora/focal-frequency-loss

    Args:
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
        super(focalfrequencyloss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def tensor2freq(self, x: torch.Tensor) -> torch.Tensor:
        # for amp dtype
        if x.dtype is not torch.float32:
            x = x.to(torch.float32)

        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert (
            h % patch_factor == 0 and w % patch_factor == 0
        ), "Patch factor should be divisible by image height and width"
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(
                    x[
                        :,
                        :,
                        i * patch_h : (i + 1) * patch_h,
                        j * patch_w : (j + 1) * patch_w,
                    ]
                )

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        freq = torch.fft.fft2(y, norm="ortho")
        freq = torch.stack([freq.real, freq.imag], -1)

        return freq

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def loss_formulation(
        self, recon_freq: torch.Tensor, real_freq: torch.Tensor, matrix: torch.Tensor = None
    ) -> torch.Tensor:
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
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = (
                    matrix_tmp
                    / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]
                )

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            "The values of spectrum weight matrix should be in the range [0, 1], "
            "but got Min: %.10f Max: %.10f"
            % (weight_matrix.min().item(), weight_matrix.max().item())
        )

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, matrix: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        """Forward function to calculate focal frequency loss.

        Args:
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
