
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import deform_conv2d, DeformConv2d

from neosr.utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights, make_layer, pixel_unshuffle, net_opt

upscale, training = net_opt()


class ResidualDenseDeformableBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseDeformableBlock, self).__init__()
        #self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv1 = nn.DeformConv2d(num_feat, num_grow_ch, 3, 1, 1)

        self.offset1 = nn.Conv2d(num_feat,
                                     2 * 3 * 3,
                                     3,
                                     1,
                                     1,
                                     bias=True)

        nn.init.constant_(self.offset1.weight, 0.)
        nn.init.constant_(self.offset1.bias, 0.)

        #self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.DeformConv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)

        self.offset2 = nn.Conv2d(num_feat + num_grow_ch,
                                     2 * 3 * 3,
                                     3,
                                     1,
                                     1,
                                     bias=True)

        nn.init.constant_(self.offset2.weight, 0.)
        nn.init.constant_(self.offset2.bias, 0.)

        #self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.DeformConv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        
        self.offset3 = nn.Conv2d(num_feat + 2 * num_grow_ch,
                                     2 * 3 * 3,
                                     3,
                                     1,
                                     1,
                                     bias=True)
        nn.init.constant_(self.offset3.weight, 0.)
        nn.init.constant_(self.offset3.bias, 0.)

        
        #self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.DeformConv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        
        self.offset4 = nn.Conv2d(num_feat + 3 * num_grow_ch,
                                     2 * 3 * 3,
                                     3,
                                     1,
                                     1,
                                     bias=True)
        nn.init.constant_(self.offset4.weight, 0.)
        nn.init.constant_(self.offset4.bias, 0.)
        
        #self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.conv5 = nn.DeformConv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.offset5 = nn.Conv2d(num_feat + 4 * num_grow_ch,
                                     2 * 3 * 3,
                                     3,
                                     1,
                                     1,
                                     bias=True)
        nn.init.constant_(self.offset5.weight, 0.)
        nn.init.constant_(self.offset5.bias, 0.)
        

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.scale_residual = nn.Parameter(torch.tensor(0.2))

        # initialization
        default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        #x1 = self.lrelu(self.conv1(x))
        #x1= deform_conv2d(x, offset=self.offset1(x), weight=self.conv1.weight,bias=self.conv1.bias,stride=1,padding=1)
        x1 = self.lrelu(self.conv1(x), offset=self.offset1(x))
        #x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        o1 = torch.cat((x, x1), 1)
        #x2= self.lrelu( deform_conv2d(o1, offset=self.offset2(o1), weight=self.conv2.weight,bias=self.conv2.bias,stride=1,padding=1) )
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)), offset=self.offset2(torch.cat((x, x1), 1))))
        #x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        o2 = torch.cat((x, x1, x2), 1)
        #x3= self.lrelu( deform_conv2d(o2, offset=self.offset3(o2), weight=self.conv3.weight,bias=self.conv3.bias,stride=1,padding=1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)), offset=self.offset3(torch.cat((x, x1, x2), 1)))
        #x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        o3 = torch.cat((x, x1, x2, x3), 1)
        #x4= self.lrelu(deform_conv2d(o3, offset=self.offset4(o3), weight=self.conv4.weight,bias=self.conv4.bias,stride=1,padding=1))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)), offset=self.offset4( torch.cat((x, x1, x2, x3), 1) )
        #x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        o4 = torch.cat((x, x1, x2, x3, x4), 1)
        #x5= self.lrelu( deform_conv2d(o4, offset=self.offset5(o4), weight=self.conv5.weight,bias=self.conv5.bias,stride=1,padding=1) )
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1), offset=self.offset5(torch.cat((x, x1, x2, x3, x4), 1) )
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * self.scale_residual + x


class RRDDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDDB, self).__init__()
        self.rdb1 = ResidualDenseDeformableBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseDeformableBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseDeformableBlock(num_feat, num_grow_ch)

        self.scale_residual = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * self.scale_residual + x


@ARCH_REGISTRY.register()
class desrgan(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=upscale, num_feat=64, num_block=23, num_grow_ch=32):
        super(desrgan, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(
            RRDDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(
            feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(
            feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
