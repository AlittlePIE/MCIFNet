#!/usr/bin/python3
# coding=utf-8

import os
import sys

sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
import dataset as dataset
from model import MCIFNet


class Test(object):
    def __init__(self, Dataset, Network, path, spath, pth):
        ## dataset
        self.cfg = Dataset.Config(datapath=path, snapshot=pth, mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0)
        ## network
        self.net = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()
        self.spath =  spath
    def save(self):
        with torch.no_grad():
            c = 0
            for image, mask, shape, name in self.loader:
                image = image.cuda().float()
                import time
                a = time.time()
                out1u, out2u, out2r, out3= self.net(image, shape)
                b = time.time()
                c = b - a + c
                out = out1u
                pred = (torch.sigmoid(out[0, 0]) * 255).cpu().numpy()
                head = self.spath
                if not os.path.exists(head):
                    os.makedirs(head)
                savename = head + '/' + name[0] + '.png'
                cv2.imwrite(savename, np.round(pred))
            print(c)
            print(250 / c)


if __name__ == '__main__':
    # Need to change!
    path = '/kaggle/input/tensorflow-great-barrier-reef/train_images/video_2/5774.jpg'
    spath = '/kaggle/output/'
    pth = '/kaggle/input/mcifnet/Net_epoch_best.pth'
    
    t = Test(dataset, MCIFNet, path, spath, pth)
    t.save()

