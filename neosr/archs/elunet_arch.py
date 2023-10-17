import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.parametrizations import spectral_norm
from neosr.utils.registry import ARCH_REGISTRY


class DoubleConv(nn.Module):
    """ [(Conv2d) => (BN) => (ReLu)] * 2 """
    
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
                # SN
                spectral_norm(nn.Conv2d(in_channels,out_channels,3,padding="same",stride=1)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                # SN
                spectral_norm(nn.Conv2d(out_channels,out_channels,3,padding="same",stride=1)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()      
            )

    def forward(self,x):
        x = self.double_conv(x)
        return x

class DownSample(nn.Module):
    """ MaxPool => DoubleConv """
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels,out_channels)
        )
    def forward(self,x):
        x  = self.down_sample(x)
        return x

class UpSample(nn.Module):
    def __init__(self,in_channels,out_channels,c:int) -> None:
        """ UpSample input tensor by a factor of `c`
                - the value of base 2 log c defines the number of upsample 
                layers that will be applied
        """
        super().__init__()
        n = 0 if c == 0 else int(math.log(c,2))

        self.upsample = nn.ModuleList(
            [nn.ConvTranspose2d(in_channels,in_channels,2,2) for i in range(n)]
        )
        # SN
        self.conv_3 = spectral_norm(nn.Conv2d(in_channels,out_channels,3,padding="same",stride=1))

    def forward(self,x):
        for layer in self.upsample:
            x = layer(x)
        x = self.conv_3(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

@ARCH_REGISTRY.register()
class elunet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n:int = 8) -> None:
        """ 
        Construct the Elu-net model, with Spectral Normalization.
        Args:
            in_channels: The number of color channels of the input image. 0:for binary 3: for RGB
            out_channels: The number of color channels of the input mask, corresponds to the number
                            of classes.Includes the background
            n: Channels size of the first CNN in the encoder layer. The bigger this value the bigger 
                the number of parameters of the model. Defaults to n = 8, which is recommended by the 
                authors of the paper.
        """
        super().__init__()
        # ------ Input convolution --------------
        self.in_conv = DoubleConv(in_channels,n)
        # -------- Encoder ----------------------
        self.down_1 = DownSample(n,2*n)
        self.down_2 = DownSample(2*n,4*n)
        self.down_3 = DownSample(4*n,8*n)
        self.down_4 = DownSample(8*n,16*n)
        
        # -------- Upsampling ------------------
        self.up_1024_512 = UpSample(16*n,8*n,2)

        self.up_512_64 = UpSample(8*n,n,8)
        self.up_512_128 = UpSample(8*n,2*n,4)
        self.up_512_256 = UpSample(8*n,4*n,2)
        self.up_512_512 = UpSample(8*n,8*n,0)

        self.up_256_64 = UpSample(4*n,n,4)
        self.up_256_128 = UpSample(4*n,2*n,2)
        self.up_256_256 = UpSample(4*n,4*n,0)

        self.up_128_64 = UpSample(2*n,n,2)
        self.up_128_128 = UpSample(2*n,2*n,0)

        self.up_64_64 = UpSample(n,n,0)
     
        # ------ Decoder block ---------------
        self.dec_4 = DoubleConv(2*8*n,8*n)
        self.dec_3 = DoubleConv(3*4*n,4*n)
        self.dec_2 = DoubleConv(4*2*n,2*n)
        self.dec_1 = DoubleConv(5*n,n)
        # ------ Output convolution

        self.out_conv = OutConv(n,out_channels)

    def forward(self,x):
        x = self.in_conv(x) # 64
        # ---- Encoder outputs
        x_enc_1 = self.down_1(x) # 128
        x_enc_2 = self.down_2(x_enc_1) # 256
        x_enc_3 = self.down_3(x_enc_2) # 512
        x_enc_4 = self.down_4(x_enc_3) # 1024
    
        # ------ decoder outputs
        x_up_1 = self.up_1024_512(x_enc_4)
        x_dec_4 = self.dec_4(torch.cat([x_up_1,self.up_512_512(x_enc_3)],dim=1))

        x_up_2 = self.up_512_256(x_dec_4)
        x_dec_3 = self.dec_3(torch.cat([x_up_2,
            self.up_512_256(x_enc_3),
            self.up_256_256(x_enc_2)
            ],
        dim=1))

        x_up_3 = self.up_256_128(x_dec_3)
        x_dec_2 = self.dec_2(torch.cat([
            x_up_3,
            self.up_512_128(x_enc_3),
            self.up_256_128(x_enc_2),
            self.up_128_128(x_enc_1)
        ],dim=1))

        x_up_4 = self.up_128_64(x_dec_2)
        x_dec_1 = self.dec_1(torch.cat([
            x_up_4,
            self.up_512_64(x_enc_3),
            self.up_256_64(x_enc_2),
            self.up_128_64(x_enc_1),
            self.up_64_64(x)
        ],dim=1))

        return self.out_conv(x_dec_1)
