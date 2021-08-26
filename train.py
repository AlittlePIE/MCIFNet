#!/usr/bin/python3
#coding=utf-8

import sys
import datetime
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
import dataset as dataset
from model import MCIFNet
import torch.nn as nn


def structure_loss(pred, mask):

    k = nn.Softmax2d()
    weit  = torch.abs(pred-mask)
    weit = k(weit)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)

    return (wbce+wiou).mean()


def val(model, epoch, save_path):
    """
    validation function
    """
    import numpy as np
    import logging
    import torch
    import torch.nn.functional as F
    from dataset import Config, Data
    from torch.utils.data import DataLoader
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    ## dataset
    cfg = Config(mode='test', datapath='/home/hhedeeplearning/share/dongbo/DATA/cod/CAMO/')
    stest_loader = Data(cfg)
    test_loader = DataLoader(stest_loader, batch_size=1, shuffle=False, num_workers=0)
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        number = 0
        for image, mask, shape, name in test_loader:
            number = 1+number
            image = image.cuda().float()
            gt = np.asarray(mask, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            res = model(image)
            res = F.upsample(res[0], size=gt.shape[1:], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.mean(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / number
        torch.save(model.state_dict(), save_path + str(epoch) + 'Net_epoch.pth')
        # writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                torch.save(model.state_dict(), save_path + str(epoch) + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))

def train(Dataset, Network):
    ## dataset
    # torch.cuda.set_device(3)
    cfg    = Dataset.Config(datapath='/home/hhedeeplearning/share/dongbo/DATA/cod/',
                            savepath='./out', mode='train', batch=16, lr=0.005, momen=0.95, decay=5e-4, epoch=32)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=8)
    ## network
    net    = Network(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    # net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    # sw             = SummaryWriter(cfg.savepath)
    global_step    = 0

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr

        for step, (image, mask) in enumerate(loader):
            image, mask = image.cuda().float(), mask.cuda().float()
            out1u, out2u, out2r, out3r = net(image)

            loss1u = structure_loss(out1u, mask)
            loss2u = structure_loss(out2u, mask)

            loss2r = structure_loss(out2r, mask)
            loss3r = structure_loss(out3r, mask)
            # loss4r = structure_loss(out2r, mask)
            # loss5r = structure_loss(out2r, mask)
            loss   = (loss1u+loss2u)+loss2r+loss3r

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ## log
            global_step += 1
            # sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            # sw.add_scalars('loss', {'loss1u':loss1u.item(), 'loss2u':loss2u.item(), 'loss2r':loss2r.item(), 'loss3r':loss3r.item(), 'loss4r':loss4r.item(), 'loss5r':loss5r.item()}, global_step=global_step)
            if step%100 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f'%(datetime.datetime.now(), global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item()))

        save_path = '/home/hhedeeplearning/share/dongbo/DATA/cod/MCIF-final1/out/'
        val(net, epoch, save_path)


if __name__=='__main__':
    best_mae=1
    best_epoch=0
    train(dataset, MCIFNet)
