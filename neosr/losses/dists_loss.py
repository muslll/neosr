from os import path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from neosr.utils.registry import LOSS_REGISTRY


class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, as_loss=True, pad_off=0) -> None:
        super().__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g /= torch.sum(g)
        self.register_buffer(
            "filter", g[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )

        if as_loss is False:
            # send to cuda
            self.filter = self.filter.cuda()

    def forward(self, input):
        input **= 2
        out = F.conv2d(
            input,
            self.filter,
            stride=self.stride,
            padding=self.padding,
            groups=input.shape[1],
        )
        return (out + 1e-12).sqrt()


@LOSS_REGISTRY.register()
class dists_loss(nn.Module):

    r"""DISTS. "Image Quality Assessment: Unifying Structure and Texture Similarity":
    https://arxiv.org/abs/2004.07728.

    Args:
    ----
        as_loss (bool): True to use as loss, False for metric.
            Default: True.
        loss_weight (float).
            Default: 1.0.
        load_weights (bool): loads pretrained weights for DISTS.
            Default: False.

    """

    def __init__(self, as_loss=True, loss_weight=1.0, load_weights=True, **kwargs) -> None:
        super().__init__()
        self.as_loss = as_loss
        self.loss_weight = loss_weight

        vgg_pretrained_features = models.vgg16(weights="DEFAULT").features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64, as_loss=as_loss))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128, as_loss=as_loss))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256, as_loss=as_loss))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512, as_loss=as_loss))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.chns = [3, 64, 128, 256, 512, 512]
        self.register_parameter(
            "alpha", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1))
        )
        self.register_parameter(
            "beta", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1))
        )
        self.alpha.data.normal_(0.1, 0.01)
        self.beta.data.normal_(0.1, 0.01)

        if load_weights:
            current_dir = osp.dirname(osp.abspath(__file__))
            model_path = osp.join(current_dir, "dists_weights.pth")
            weights = torch.load(model_path) if osp.exists(model_path) else None

            self.alpha.data = weights["alpha"]
            self.beta.data = weights["beta"]

            if as_loss is False:
                # send to cuda
                self.alpha.data = self.alpha.data.cuda()
                self.beta.data = self.beta.data.cuda()

    def forward_once(self, x):
        h = x
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    # @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, x, y):
        feats0 = self.forward_once(x)
        feats1 = self.forward_once(y)
        dist1 = 0
        dist2 = 0
        c1 = 1e-6
        c2 = 1e-6

        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha / w_sum, self.chns, dim=1)
        beta = torch.split(self.beta / w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2, 3], keepdim=True)
            y_mean = feats1[k].mean([2, 3], keepdim=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)
            dist1 += (alpha[k] * S1).sum(1, keepdim=True)

            x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (feats0[k] * feats1[k]).mean(
                [2, 3], keepdim=True
            ) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 += (beta[k] * S2).sum(1, keepdim=True)

        if self.as_loss:
            out = 1 - (dist1 + dist2).mean()
            out *= self.loss_weight
        else:
            out = 1 - (dist1 + dist2).squeeze()

        return out
