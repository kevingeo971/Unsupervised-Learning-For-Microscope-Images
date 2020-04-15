""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, only_encode=False, bilinear=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.only_encode = only_encode

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
    
        self.up1_low = Up(1024, 256, bilinear)
        self.up2_low = Up(512, 128, bilinear)
        self.up3_low = Up(256, 64, bilinear)
        self.up4_low = Up(128, 64, bilinear)
        self.outc_low = OutConv(64, out_channels)

        self.up1_high = Up(1024, 256, bilinear)
        self.up2_high = Up(512, 128, bilinear)
        self.up3_high = Up(256, 64, bilinear)
        self.up4_high = Up(128, 64, bilinear)
        self.outc_high = OutConv(64, out_channels)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print("\n\n", x5.shape, "\n\n")
        if self.only_encode:
            return x5
        x_low = self.up1_low(x5, x4)
        x_low = self.up2_low(x_low, x3)
        x_low = self.up3_low(x_low, x2)
        x_low = self.up4_low(x_low, x1)
        logits_low = self.outc_low(x_low)
        
        x_high = self.up1_high(x5, x4)
        x_high = self.up2_high(x_high, x3)
        x_high = self.up3_high(x_high, x2)
        x_high = self.up4_high(x_high, x1)
        logits_high = self.outc_high(x_high)
        
        #logits = torch.stack((logits_low,logits_high)).squeeze().permute(1,0,2,3)

        return logits_low, logits_high
