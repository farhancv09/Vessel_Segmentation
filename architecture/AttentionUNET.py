import math
import torch
from torch import nn, cat, add
import numpy as np
import torch.nn.functional as F


# Attention UNet3D

class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(3, 3, 3), padding_size=(1, 1, 1), init_stride=(1, 1, 1)):
        super(UnetConv3, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                   nn.GroupNorm(4, out_size),
                                   nn.ReLU(inplace=True), )
        self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                   nn.GroupNorm(4, out_size),
                                   nn.ReLU(inplace=True), )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetGridGatingSignal(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(1, 1, 1)):
        super(UnetGridGatingSignal, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size, (1, 1, 1), (0, 0, 0)),
                                   nn.GroupNorm(4, out_ch),
                                   nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class GridAttentionBlock3D(nn.Module):
    def __init__(self, x_ch, g_ch, sub_sample_factor=(2, 2, 2)):
        super(GridAttentionBlock3D, self).__init__()

        self.W = nn.Sequential(
            nn.Conv3d(x_ch,
                      x_ch,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.GroupNorm(4, x_ch))
        self.theta = nn.Conv3d(x_ch,
                               x_ch,
                               kernel_size=sub_sample_factor,
                               stride=sub_sample_factor,
                               padding=0,
                               bias=False)
        self.phi = nn.Conv3d(g_ch,
                             x_ch,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)
        self.psi = nn.Conv3d(x_ch,
                             out_channels=1,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.upsample(self.phi(g),
                           size=theta_x_size[2:],
                           mode='trilinear')

        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = F.sigmoid(self.psi(f))
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode='trilinear')
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y


class UnetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetUp3, self).__init__()

        self.conv = UnetConv3(in_size, out_size)
        self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


    
    
    
    
class AttentionUNet3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(AttentionUNet3D, self).__init__()

        self.encoderconv1 = UnetConv3(n_channels, 32)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.encoderconv2 = UnetConv3(32, 64)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.encoderconv3 = UnetConv3(64, 128)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.encoderconv4 = UnetConv3(128, 256)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(256, 512)
        self.gating = UnetGridGatingSignal(512, 256, kernel_size=(1, 1, 1))

        self.attentionblock4 = GridAttentionBlock3D(256, 256)
        self.attentionblock3 = GridAttentionBlock3D(128, 256)
        self.attentionblock2 = GridAttentionBlock3D(64, 256)

        self.up_concat4 = UnetUp3(512, 256)
        self.up_concat3 = UnetUp3(256, 128)
        self.up_concat2 = UnetUp3(128, 64)
        self.up_concat1 = UnetUp3(64, 32)

        self.out_conv = nn.Conv3d(32, n_classes, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_input):
        x_en1 = self.encoderconv1(x_input)
        pool1 = self.maxpool1(x_en1)

        x_en2 = self.encoderconv2(pool1)
        pool2 = self.maxpool2(x_en2)

        x_en3 = self.encoderconv3(pool2)
        pool3 = self.maxpool3(x_en3)

        x_en4 = self.encoderconv4(pool3)
        pool4 = self.maxpool4(x_en4)

        center = self.center(pool4)
        gating = self.gating(center)

        att4 = self.attentionblock4(x_en4, gating)
        att3 = self.attentionblock3(x_en3, gating)
        att2 = self.attentionblock2(x_en2, gating)

        up4 = self.up_concat4(att4, center)
        up3 = self.up_concat3(att3, up4)
        up2 = self.up_concat2(att2, up3)
        up1 = self.up_concat1(x_en1, up2)

        x = self.out_conv(up1)

        x = self.softmax(x,dim=1)
        return x
