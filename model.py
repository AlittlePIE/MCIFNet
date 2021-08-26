#!/usr/bin/python3
#coding=utf-8

###################
#MCIF-Net
##################

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

def weight_init(module):
    for n, m in module.named_children():
        # print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('resnet50.pth'), strict=False)


class MIF(nn.Module):
    def __init__(self):
        super(MIF, self).__init__()
        #D1
        self.conv1v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1v   = nn.BatchNorm2d(64)
        #L1
        self.conv2h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2h   = nn.BatchNorm2d(64)
        #L2
        self.conv2v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2v   = nn.BatchNorm2d(64)
        #last CBR
        self.conv3v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3v   = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=1)


    def forward(self, left, down):
        #left:low-level
        #down:high-level

        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

        out1v = F.relu(self.bn1v(self.conv1v(down )), inplace=True)
        avg_out1= torch.mean(out1v, dim=1, keepdim=True)
        max_out1, _ = torch.max(out1v, dim=1, keepdim=True)

        out1h = F.relu(self.bn2h(self.conv2h(left )), inplace=True)
        avg_out2 = torch.mean(out1h, dim=1, keepdim=True)
        max_out2, _ = torch.max(out1h, dim=1, keepdim=True)

        avg_out1_1 = avg_out1 * F.interpolate(max_out2, size=avg_out1.size()[2:], mode='bilinear')
        max_out1_1 = max_out1 * F.interpolate(avg_out2, size=max_out1.size()[2:], mode='bilinear')
        avg_out2_1 = avg_out2 * F.interpolate(max_out1, size=avg_out2.size()[2:], mode='bilinear')
        max_out2_1 = max_out2 * F.interpolate(avg_out1, size=max_out2.size()[2:], mode='bilinear')

        scale1 = torch.cat([avg_out1_1, max_out1_1], dim=1)
        scale1 = self.conv1(scale1)
        scale1 = F.interpolate(scale1, size=out1v.size()[2:], mode='bilinear')
        s = out1v * self.sigmoid1(scale1) + out1v

        scale2 = torch.cat([avg_out2_1, max_out2_1], dim=1)
        scale2 = self.conv2(scale2)
        scale2 = F.interpolate(scale2, size=out1h.size()[2:], mode='bilinear')
        out1h = out1h * self.sigmoid2(scale2) + out1h
        out2v = F.relu(self.bn2v(self.conv2v(out1h)), inplace=True)

        if out2v.size()[2:] != s.size()[2:]:
            s = F.interpolate(s, size=out2v.size()[2:], mode='bilinear')

        fuse  = s*out2v
        fuse = F.relu(self.bn3v(self.conv3v(fuse)), inplace=True)

        return fuse

    def initialize(self):
        weight_init(self)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.mif3  = MIF()
        self.mif2  = MIF()
        self.mif1  = MIF()

    def forward(self, Xd1, Xd2, Xd3, Xd4, fback=None):
        d1 = F.interpolate(Xd2, size=Xd1.size()[2:], mode='bilinear')
        Xm1 = d1 * Xd1

        d2 = F.interpolate(Xd3, size=Xd2.size()[2:], mode='bilinear')
        Xm2 = d2*Xd2

        d2 = F.interpolate(Xd4, size=Xd3.size()[2:], mode='bilinear')
        Xm3 = d2 * Xd3

        Xa3 = self.mif3(Xm3 , Xd4)
        Xa2 = self.mif2(Xm2, Xa3)
        xa1 = self.mif1(Xm1, Xa2)

        return Xa2, Xa3, Xd4, xa1

    def initialize(self):
        weight_init(self)



class DMC(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(DMC, self).__init__()
        self.relu = nn.ReLU(True)

        self.conv_res = BasicConv2d(in_channel, in_channel, 3)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),

        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            # BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.branch21 = nn.Sequential(BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5))
        self.branch31 = nn.Sequential( BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7))

        self.conv_cat = BasicConv2d(2*out_channel, out_channel, 3, padding=1)


    def forward(self, x):

        self.conv_res(x)

        x2 = self.branch2(x)
        x1 = self.branch3(x)+x2

        x2 = self.branch21(x1)
        x3 = self.branch31(x1)

        x_cat = self.conv_cat(torch.cat((x2, x3), 1))
        y = self.relu(x_cat)

        return y

    def initialize(self):
        weight_init(self)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    def initialize(self):
        weight_init(self)

class MCIFNet(nn.Module):
    def __init__(self, cfg):
        super(MCIFNet, self).__init__()
        self.cfg      = cfg
        self.bkbone   = ResNet()
        self.dmc4 = DMC(2048, 64)
        self.dmc3 = DMC(1024, 64)
        self.dmc2 = DMC(512, 64)
        self.dmc1 = DMC(256, 64)

        self.decoder1 = Decoder()

        self.linearp1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.initialize()

    def forward(self, x, shape=None):
        #four outputs of backbone
        x1, x2, x3, x4 = self.bkbone(x)

        Xd1, Xd2, Xd3, Xd4= self.dmc1(x1), self.dmc2(x2), self.dmc3(x3), self.dmc4(x4)

        Xa2, Xa3, Xd4, xa1= self.decoder1(Xd1, Xd2, Xd3, Xd4)

        shape = x.size()[2:] if shape is None else shape
        xa1 = F.interpolate(self.linearp1(xa1), size=shape, mode='bilinear')
        Xa2 = F.interpolate(self.linearr3(Xa2), size=shape, mode='bilinear')
        Xa3 = F.interpolate(self.linearr4(Xa3), size=shape, mode='bilinear')
        Xd4 = F.interpolate(self.linearr5(Xd4), size=shape, mode='bilinear')

        return xa1, Xa2, Xa3, Xd4

    # , map_location = 'gpu'
    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot,map_location='cuda:0'))
        else:
            weight_init(self)

if __name__ == '__main__':
    # cfg    = Dataset.Config(datapath='/media/zdc/dz/data/salience/COD10K',
    #                         savepath='./out', mode='train', batch=1, lr=0.005, momen=0.95, decay=5e-4, epoch=32)
    model = MCIFNet()
    input = torch.rand(1,3,352,352)
    output = model(input)
    print(output)