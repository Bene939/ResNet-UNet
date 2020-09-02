import torch.nn as nn
import torch

###########################################################################
# Question 1 : Implement the UNet model code.
# Understand architecture of the UNet in practice lecture 15 -> slides 5-6 (30 points)

def convl(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),  # 3은 kernel size
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()

        ########## fill in the blanks (Hint : check out the channel size in practice lecture 15 ppt slides 5-6)
        self.convDown1 = convl(3, in_channels)
        self.convDown2 = convl(64, 128)
        self.convDown3 = convl(128, 256)
        self.convDown4 = convl(256, 512)
        self.convDown5 = convl(512, 1024)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.convUp4 = convl(1536, 512)
        self.convUp3 = convl(768, 256)
        self.convUp2 = convl(384, 128)
        self.convUp1 = convl(192, 64)
        self.convUp_fin = nn.Conv2d(64, out_channels, 1)



    def forward(self, x):
        conv1 = self.convDown1(x)
        x = self.maxpool(conv1)
        conv2 = self.convDown2(x)
        x = self.maxpool(conv2)
        conv3 = self.convDown3(x)
        x = self.maxpool(conv3)
        conv4 = self.convDown4(x)
        x = self.maxpool(conv4)
        conv5 = self.convDown5(x)
        x = self.upsample(conv5)
        #######fill in here ####### hint : concatenation (Practice Lecture slides 6p)
        x = torch.cat([x, conv4], dim=1)
        x = self.convUp4(x)
        x = self.upsample(x)
        #######fill in here ####### hint : concatenation (Practice Lecture slides 6p)
        x = torch.cat([x, conv3], dim=1)
        x = self.convUp3(x)
        x = self.upsample(x)
        #######fill in here ####### hint : concatenation (Practice Lecture slides 6p)
        x = torch.cat([x, conv2], dim=1)
        x = self.convUp2(x)
        x = self.upsample(x)
        #######fill in here ####### hint : concatenation (Practice Lecture slides 6p)
        x = torch.cat([x, conv1], dim=1)
        x = self.convUp1(x)
        out = self.convUp_fin(x)

        return out