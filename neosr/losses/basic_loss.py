import torch
import torch.fft
import torchvision.models
from torch import nn
from torch.nn import functional as F

from neosr.archs.vgg_arch import VGGFeatureExtractor
from neosr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss
from neosr.utils.color_util import rgb_to_uv

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def huber_loss(pred, target, delta=1.0):
    return F.huber_loss(pred, target, delta=1.0)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(
                f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(
                f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


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

    def __init__(self, loss_weight=1.0, reduction='mean', delta=1.0):
        super(HuberLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.delta = delta

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * huber_loss(pred, target, weight, delta=self.delta, reduction=self.reduction)


class Resnet_loss(nn.Module):
    def __init__(self, cnn, feature_layers=3):
        super(Resnet_loss, self).__init__()
        self.conv1 = cnn.conv1
        self.bn1 = cnn.bn1
        self.relu = cnn.relu
        self.maxpool = cnn.maxpool
        self.feature_layers = feature_layers
        assert feature_layers <= 4
        self.blocks = nn.ModuleList()
        for _, layer in zip(range(4 + feature_layers), cnn.children()):
            if isinstance(layer, nn.Sequential):
                self.blocks.append(layer)

    def forward(self, predict_features, target_features, weights=[1]):
        if len(weights) == 1:
            weights = weights * self.feature_layers
        x = predict_features
        x = self.bn1(self.conv1(x))
        x = self.maxpool(self.relu(x))

        y = target_features
        y = self.bn1(self.conv1(y))
        y = self.maxpool(self.relu(y))

        losses = []
        L1_loss = nn.L1Loss().cuda()
        for block in self.blocks:
            x = block(x)
            y = block(y)
            losses.append(L1_loss(x, y))
        total_loss = 0
        for weight, loss in zip(weights, losses):
            total_loss += loss * weight
        return total_loss


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

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 perceptual_type='dual',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 perceptual_patch_weight=0.1,
                 style_weight=0.,
                 criterion='patch',
                 perceptual_kernels=[4,8],
                 use_std_to_force=True):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.patch_weights = perceptual_patch_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.perceptual_kernels = perceptual_kernels
        self.use_std_to_force = use_std_to_force
        self.perceptual_type = perceptual_type
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        if self.perceptual_type == 'dual':
            resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            self.resnet_loss = Resnet_loss(resnet).cuda()

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'huber':
            self.criterion = torch.nn.HuberLoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        elif self.criterion_type == 'patch':
            self.criterion = torch.nn.L1Loss()
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        if self.perceptual_type == 'dual':
            percep_loss1 = self.resnet_loss(x, gt.detach())

        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        if self.perceptual_type != 'dual':
            # calculate perceptual loss
            if self.perceptual_weight > 0:
                percep_loss = 0
                for k in x_features.keys():
                    if self.criterion_type == 'fro':
                        percep_loss += torch.norm(
                            x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                    elif self.criterion_type == 'patch':
                        if self.patch_weights == 0:
                            percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
                        else:
                            percep_loss += self.patch(x_features[k], gt_features[k], self.use_std_to_force) * \
                                            self.layer_weights[k] * self.patch_weights + self.criterion(x_features[k],
                                                                                                        gt_features[
                                                                                                            k]) * \
                                            self.layer_weights[k]
                    else:
                        percep_loss += self.criterion(
                            x_features[k], gt_features[k]) * self.layer_weights[k]
                percep_loss *= self.perceptual_weight
            else:
                percep_loss = None
        elif self.perceptual_type == 'dual':
            if self.perceptual_weight > 0:
                percep_loss2 = 0
                for k in x_features.keys():
                    if self.criterion_type == 'fro':
                        percep_loss2 += torch.norm(
                            x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                    elif self.criterion_type == 'patch':
                        if self.patch_weights == 0:
                            percep_loss2 += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
                        else:
                            percep_loss2 += self.patch(x_features[k], gt_features[k], self.use_std_to_force) * \
                                           self.layer_weights[k] * self.patch_weights + self.criterion(x_features[k],
                                                                                                       gt_features[
                                                                                                           k]) * \
                                           self.layer_weights[k]
                    else:
                        percep_loss2 += self.criterion(
                            x_features[k], gt_features[k]) * self.layer_weights[k]
                percep_loss2 *= self.perceptual_weight
            else:
                percep_loss2 = None

            a = percep_loss1.item()
            b = percep_loss2.item()
            mu = (1 / 0.5)
            DP_loss = mu * (b / a) * percep_loss1 + percep_loss2
        else:
            percep_loss = None
            percep_loss1 = None
            percep_loss2 = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        if self.perceptual_type != 'dual':
            percep_loss1 = None
            percep_loss2 = None
            return percep_loss, style_loss, percep_loss1, percep_loss2
        else:
            return DP_loss, style_loss, percep_loss1, percep_loss2

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

    def patch(self, x, gt, use_std_to_force):
        loss = 0.
        for _kernel in self.perceptual_kernels:
            _patchkernel3d = PatchesKernel3D(_kernel, _kernel//2).to('cuda:0')   # create instance
            x_trans = _patchkernel3d(x)
            gt_trans = _patchkernel3d(gt)
            x_trans = x_trans.reshape(-1, x_trans.shape[-1])
            gt_trans = gt_trans.reshape(-1, gt_trans.shape[-1])
            dot_x_y = torch.einsum('ik,ik->i', x_trans, gt_trans)
            if use_std_to_force == False:
                cosine0_x_y = torch.div(torch.div(dot_x_y, torch.sqrt(torch.sum(x_trans ** 2, dim=1))), torch.sqrt(torch.sum(gt_trans ** 2, dim=1)))
                loss = loss + torch.mean(1-cosine0_x_y) # y = 1-x
            else:
                dy = torch.std(gt_trans, dim=1)
                cosine_x_y = torch.div(torch.div(dot_x_y, torch.sqrt(torch.sum(x_trans ** 2, dim=1))), torch.sqrt(torch.sum(gt_trans ** 2, dim=1)))
                cosine_x_y_d = torch.mul((1-cosine_x_y), dy) # y = (1-x)dy
                loss = loss + torch.mean(cosine_x_y_d)
        return loss


class PatchesKernel3D(nn.Module):
    def __init__(self, kernelsize, kernelstride, kernelpadding=0):
        super(PatchesKernel3D, self).__init__()
        kernel = torch.eye(kernelsize ** 2).\
            view(kernelsize ** 2, 1, kernelsize, kernelsize)
        kernel = kernel.clone().detach()
        self.weight = nn.Parameter(data=kernel,
                                   requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(kernelsize ** 2),
                                 requires_grad=False)
        self.kernelsize = kernelsize
        self.stride = kernelstride
        self.padding = kernelpadding

    def forward(self, x):
        batchsize = x.shape[0]
        channels = x.shape[1]
        x = x.reshape(batchsize*channels, x.shape[-2], x.shape[-1]).unsqueeze(1)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        x = x.permute(0, 2, 3, 1).reshape(batchsize, channels, -1, self.kernelsize ** 2).permute(0, 2, 1, 3)
        return x


@LOSS_REGISTRY.register()
class colorloss(nn.Module):
    def __init__(self, criterion='l1', loss_weight=1.0):
        super(colorloss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'huber':
            self.criterion = torch.nn.HuberLoss()
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported.')

    def forward(self, input, target):
        input_uv = rgb_to_uv(input)
        target_uv = rgb_to_uv(target)
        return self.criterion(input_uv, target_uv) * self.loss_weight

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
    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=True,
                 log_matrix=False, batch_matrix=False):
        super(focalfrequencyloss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def tensor2freq(self, x):

        # for amp dtype
        if x.dtype is not torch.float32:
            x = x.to(torch.float32)

        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        freq = torch.fft.fft2(y, norm='ortho')
        freq = torch.stack([freq.real, freq.imag], -1)

        return freq

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, pred, target, matrix=None, **kwargs):
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


@LOSS_REGISTRY.register()
class patchloss(nn.Module):
    """Define patch loss

    Args:
        kernel_sizes (list): add (x, y) in the list.
        loss_weight (float): Loss weight. Default: 1.0.
    """
    def __init__(self, kernel_sizes=[2, 4], loss_weight=1.0):
        super(patchloss, self).__init__()
        self.kernels = kernel_sizes
        self.loss_weight = loss_weight

    def forward(self, preds, labels):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            labels (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        # from basicsr.utils.matlab_functions import rgb2ycbcr

        # PIL proves that the data format is RGB
        # from PIL import Image
        # for i_ in range(16):
        #     label = labels[i_,:,:,:] # torch [3, 96, 96]
        #     label9 = (label * 255).detach().cpu().numpy().astype(np.uint8).transpose(1,2,0) # [96, 96, 3]
        #     im = Image.fromarray(label9)
        #     im.save(str(i_)+'.png')

        # for j_ in range(16):
        #     label = labels[j_,:,:,:]
        #     label9 = (label * 255).detach().cpu().numpy().astype(np.uint8).transpose(1,2,0) # [96, 96, 3]
        #     _a  =  np.expand_dims(label9[:,:,2],axis=-1)
        #     _b  =  np.expand_dims(label9[:,:,1],axis=-1)
        #     _c  =  np.expand_dims(label9[:,:,0],axis=-1)
        #     labelx = np.concatenate([_a,_b,_c], axis = -1)
        #     im = Image.fromarray(labelx)
        #     im.save(str(j_)+'xx.png')

        # labels -> [16,3,96,96]
        # label0 = labels[1,:,:,:] # label0 -> [3,96,96]
        # label3 = label0.unsqueeze(-1).permute(3,1,2,0).squeeze() # label3 -> [96,96,3]
        # label4 = label0.unsqueeze(-1).transpose(0,3).squeeze() # label4 -> [96,96,3]
        # label6 = (16. + (65.481 * label3[:, :, 0] + 128.553 * label3[:, :, 1] + 24.966 * label3[:, :, 2]))/255.  # [96, 96]

        # label1 = label0.detach().cpu().numpy().astype(np.float32) # [3, 96, 96]
        # label2 = reorder_image(label1, input_order='CHW')  # [96, 96, 3]
        # label5 = rgb2ycbcr(label2, y_only=True)  # [96, 96]
        # # Label5 and Label6 are equal in value.

        # labels7 = (16. + (65.481 * labels[:, 0, :, :] + 128.553 * labels[:, 1, :, :] + 24.966 * labels[:, 2, :, :]))/255. # [16, 96, 96]
        # # Only one line can do this:
        # # labels7[1] == label6 == label5

        preds = (16. + (65.481 * preds[:, 0, :, :] + 128.553 * preds[:, 1, :, :] + 24.966 * preds[:, 2, :, :]))/255.
        labels = (16. + (65.481 * labels[:, 0, :, :] + 128.553 * labels[:, 1, :, :] + 24.966 * labels[:, 2, :, :]))/255.
        preds = preds.unsqueeze(1)
        labels = labels.unsqueeze(1)
        loss = 0.

        for _kernel in self.kernels:
            _patchkernel = PatchesKernel(_kernel, _kernel//2 + 1).to('cuda:0')           # create instance
            preds_trans = _patchkernel(preds)                                          # [N, patch_num, patch_len ** 2]
            labels_trans = _patchkernel(labels)                                        # [N, patch_num, patch_len ** 2]
            preds_trans = preds_trans.reshape(-1, preds_trans.shape[-1])               # [N * patch_num, patch_len ** 2]
            labels_trans = labels_trans.reshape(-1, labels_trans.shape[-1])            # [N * patch_num, patch_len ** 2]
            x = torch.clamp(preds_trans, 0.000001, 0.999999)
            y = torch.clamp(labels_trans, 0.000001, 0.999999)
            dot_x_y = torch.einsum('ik,ik->i',x,y)                                     # [N * patch_num]
            pearson_x_y = torch.mean(torch.div(torch.div(dot_x_y, torch.sqrt(torch.sum(x ** 2, dim=1))), torch.sqrt(torch.sum(y ** 2, dim=1))))
            loss = loss + torch.exp(-pearson_x_y) # y = e^(-x)
            # loss = loss - pearson_x_y # y = - x

        return loss * self.loss_weight

class PatchesKernel(nn.Module):
    def __init__(self, kernelsize, kernelstride, kernelpadding=0):
        super(PatchesKernel, self).__init__()
        kernel = torch.eye(kernelsize ** 2).\
            view(kernelsize ** 2, 1, kernelsize, kernelsize)
        kernel = kernel.clone().detach()
        self.weight = nn.Parameter(data=kernel,
                                   requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(kernelsize ** 2),
                                 requires_grad=False)
        self.kernelsize = kernelsize
        self.stride = kernelstride
        self.padding = kernelpadding

    def forward(self, x):
        batchsize = x.shape[0]
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        x = x.permute(0, 2, 3, 1).reshape(batchsize, -1, self.kernelsize ** 2)
        return x


@LOSS_REGISTRY.register()
class patchloss3d(nn.Module):
    """Define patch loss

    Args:
        kernel_sizes (list): add (4,2), (8,4), (16,8), (32,16) or (64,32) in the list.
        loss_weight (float): Loss weight. Default: 1.0.
    """
    def __init__(self, kernel_sizes=[2, 4], loss_weight=1.0):
        super(patchloss3d, self).__init__()
        self.kernels = kernel_sizes
        self.loss_weight = loss_weight

    def forward(self, preds, labels):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            labels (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        loss = 0.
        for _kernel in self.kernels:
            _patchkernel = PatchesKernel3D(_kernel, _kernel//2 + 1).to('cuda:0') # create instance
            preds_trans = _patchkernel(preds)                                  # [N, patch_num, channels, patch_len ** 2]
            labels_trans = _patchkernel(labels)                                # [N, patch_num, channels, patch_len ** 2]
            preds_trans = preds_trans.reshape(-1, preds_trans.shape[-1])       # [N * patch_num * channels, patch_len ** 2]
            labels_trans = labels_trans.reshape(-1, labels_trans.shape[-1])    # [N * patch_num * channels, patch_len ** 2]
            x = torch.clamp(preds_trans, 0.000001, 0.999999)
            y = torch.clamp(labels_trans, 0.000001, 0.999999)
            dot_x_y = torch.einsum('ik,ik->i',x,y)                             # [N * patch_num]
            cosine0_x_y = torch.mean(torch.div(torch.div(dot_x_y, torch.sqrt(torch.sum(x ** 2, dim=1))), torch.sqrt(torch.sum(y ** 2, dim=1))))
            loss = loss + (1 - cosine0_x_y) # y = 1 - x
        return loss * self.loss_weight

@LOSS_REGISTRY.register()
class patchloss3dxd(nn.Module):
    """Define patch loss

    Args:
        kernel_sizes (list): add (x, y) in the list.
        loss_weight (float): Loss weight. Default: 1.0.
    """
    def __init__(self, kernel_sizes=[2, 4], loss_weight=1.0, use_std_to_force=True):
        super(patchloss3dxd, self).__init__()
        self.kernels = kernel_sizes
        self.loss_weight = loss_weight
        self.use_std_to_force = use_std_to_force

    def forward(self, preds, labels):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            labels (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        loss = 0.
        for _kernel in self.kernels:
            _patchkernel = PatchesKernel3D(_kernel, _kernel//2 + 1).to('cuda:0') # create instance
            preds_trans = _patchkernel(preds)                                  # [N, patch_num, channels, patch_len ** 2]
            labels_trans = _patchkernel(labels)                                # [N, patch_num, channels, patch_len ** 2]
            preds_trans = preds_trans.reshape(-1, preds_trans.shape[-1])       # [N * patch_num * channels, patch_len ** 2]
            labels_trans = labels_trans.reshape(-1, labels_trans.shape[-1])    # [N * patch_num * channels, patch_len ** 2]
            x = torch.clamp(preds_trans, 0.000001, 0.999999)
            y = torch.clamp(labels_trans, 0.000001, 0.999999)
            dot_x_y = torch.einsum('ik,ik->i',x,y)
            if self.use_std_to_force == False:
                cosine0_x_y = torch.div(torch.div(dot_x_y, torch.sqrt(torch.sum(x ** 2, dim=1))), torch.sqrt(torch.sum(y ** 2, dim=1)))
                loss = loss + torch.mean((1-cosine0_x_y)) # y = 1-x
            else:
                dy = torch.std(labels_trans*10, dim=1)
                cosine_x_y = torch.div(torch.div(dot_x_y, torch.sqrt(torch.sum(x ** 2, dim=1))), torch.sqrt(torch.sum(y ** 2, dim=1)))
                cosine_x_y_d = torch.mul((1-cosine_x_y), dy) # y = (1-x) dy
                loss = loss + torch.mean(cosine_x_y_d)
        return loss * self.loss_weight


@LOSS_REGISTRY.register()
class gradientvarianceloss(nn.Module):
    """Class for calculating GV loss between two RGB images
       :parameter
       patch_size : int, scalar, size of the patches extracted from the gt and predicted images
       cpu : bool,  whether to run calculation on cpu or gpu
        """
    def __init__(self, patch_size=8, cpu=False, loss_weight=1.0):
        super(gradientvarianceloss, self).__init__()
        self.patch_size = patch_size
        self.loss_weight = loss_weight
        # Sobel kernel for the gradient map calculation
        self.kernel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        self.kernel_y = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0)
        if not cpu:
            self.kernel_x = self.kernel_x.cuda()
            self.kernel_y = self.kernel_y.cuda()
        # operation for unfolding image into non overlapping patches
        self.unfold = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size)

    def forward(self, output, target):
        # converting RGB image to grayscale
        gray_output = 0.2989 * output[:, 0:1, :, :] + 0.5870 * output[:, 1:2, :, :] + 0.1140 * output[:, 2:, :, :]
        gray_target = 0.2989 * target[:, 0:1, :, :] + 0.5870 * target[:, 1:2, :, :] + 0.1140 * target[:, 2:, :, :]

        # calculation of the gradient maps of x and y directions
        gx_target = F.conv2d(gray_target, self.kernel_x, stride=1, padding=1)
        gy_target = F.conv2d(gray_target, self.kernel_y, stride=1, padding=1)
        gx_output = F.conv2d(gray_output, self.kernel_x, stride=1, padding=1)
        gy_output = F.conv2d(gray_output, self.kernel_y, stride=1, padding=1)

        # unfolding image to patches
        gx_target_patches = self.unfold(gx_target)
        gy_target_patches = self.unfold(gy_target)
        gx_output_patches = self.unfold(gx_output)
        gy_output_patches = self.unfold(gy_output)

        # calculation of variance of each patch
        var_target_x = torch.var(gx_target_patches, dim=1)
        var_output_x = torch.var(gx_output_patches, dim=1)
        var_target_y = torch.var(gy_target_patches, dim=1)
        var_output_y = torch.var(gy_output_patches, dim=1)

        # loss function as a MSE between variances of patches extracted from gradient maps
        gradvar_loss = F.mse_loss(var_target_x, var_output_x) + F.mse_loss(var_target_y, var_output_y)
        loss = gradvar_loss * self.loss_weight

        return loss

@LOSS_REGISTRY.register()
class bbl(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, ksize=3, pad=0, stride=3, dist_norm='l2', criterion='l1', loss_weight=1.0):
        super(bbl, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ksize = ksize
        self.pad = pad
        self.stride = stride
        self.dist_norm = dist_norm
        self.loss_weight = loss_weight

        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss(reduction='mean')
        elif criterion == 'l2':
            self.criterion = torch.nn.MSEloss(reduction='mean')
        elif self.criterion_type == 'huber':
            self.criterion = torch.nn.HuberLoss()
        else:
            raise NotImplementedError('%s criterion has not been supported.' % criterion)

    def pairwise_distance(self, x, y=None):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a BxNxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        if y is None:
            dist = dist - torch.diag(dist.diag())

        return torch.clamp(dist, 0.0, float("inf"))

    def batch_pairwise_distance(self, x, y=None):
        '''
        Input: x is a BxNxd matrix
               y is an optional BxMxd matirx
        Output: dist is a BxNxM matrix where dist[b,i,j] is the square norm between x[b,i,:] and y[b,j,:]
                if y is not given then use 'y=x'.
        i.e. dist[b,i,j] = ||x[b,i,:]-y[b,j,:]||^2
        '''
        B, N, d = x.size()
        if self.dist_norm == 'l1':
            x_norm = x.view(B, N, 1, d)
            if y is not None:
                y_norm = y.view(B, 1, -1, d)
            else:
                y_norm = x.view(B, 1, -1, d)
            dist = torch.abs(x_norm - y_norm).sum(dim=3)
        elif self.dist_norm == 'l2':
            x_norm = (x ** 2).sum(dim=2).view(B, N, 1)
            if y is not None:
                M = y.size(1)
                y_t = torch.transpose(y, 1, 2)
                y_norm = (y ** 2).sum(dim=2).view(B, 1, M)
            else:
                y_t = torch.transpose(x, 1, 2)
                y_norm = x_norm.view(B, 1, N)

            dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
            # Ensure diagonal is zero if x=y
            if y is None:
                dist = dist - torch.diag_embed(torch.diagonal(dist, dim1=-2, dim2=-1), dim1=-2, dim2=-1)
            dist = torch.clamp(dist, 0.0, float("inf"))
            # dist = torch.sqrt(torch.clamp(dist, 0.0, np.inf) / d)
        else:
            raise NotImplementedError('%s norm has not been supported.' % self.dist_norm)

        return dist

    def forward(self, x, gt):
        p1 = F.unfold(x, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        B, C, H = p1.size()
        p1 = p1.permute(0, 2, 1).contiguous() # [B, H, C]

        p2 = F.unfold(gt, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        p2 = p2.permute(0, 2, 1).contiguous() # [B, H, C]

        gt_2 = F.interpolate(gt, scale_factor=0.5, mode='bicubic', align_corners = False)
        p2_2 = F.unfold(gt_2, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        p2_2 = p2_2.permute(0, 2, 1).contiguous() # [B, H, C]

        gt_4 = F.interpolate(gt, scale_factor=0.25, mode='bicubic',align_corners = False)
        p2_4 = F.unfold(gt_4, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        p2_4 = p2_4.permute(0, 2, 1).contiguous() # [B, H, C]
        p2_cat = torch.cat([p2, p2_2, p2_4], 1)

        score1 = self.alpha * self.batch_pairwise_distance(p1, p2_cat)
        score = score1 + self.beta * self.batch_pairwise_distance(p2, p2_cat) # [B, H, H]

        weight, ind = torch.min(score, dim=2) # [B, H]
        index = ind.unsqueeze(-1).expand([-1, -1, C]) # [B, H, C]
        sel_p2 = torch.gather(p2_cat, dim=1, index=index) # [B, H, C]

        loss = self.criterion(p1, sel_p2)

        return loss * self.loss_weight