import sys
sys.path.append('/home/anirudhan/project/image-fusion')


import numpy as np
from skimage.metrics import structural_similarity
from config.jax_train_config import get_default_configs

config = get_default_configs()

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.ndimage import distance_transform_edt as distance
# can find here: https://github.com/CoinCheung/pytorch-loss/blob/af876e43218694dc8599cc4711d9a5c5e043b1b2/label_smooth.py
# from .label_smooth import LabelSmoothSoftmaxCEV1 as LSSCE
from torchvision import transforms
from functools import partial
from operator import itemgetter

# Tools
def kl_div(a,b): # q,p
    return F.softmax(b, dim=1) * (F.log_softmax(b, dim=1) - F.log_softmax(a, dim=1))   

def one_hot2dist(seg):
    res = np.zeros_like(seg)
    for i in range(len(seg)):
        posmask = seg[i].astype(np.bool_)
        if posmask.any():
            negmask = ~posmask
            res[i] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res

def class2one_hot(seg, C):
    seg = seg.unsqueeze(dim=0) if len(seg.shape) == 2 else seg
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    return res

# Active Boundary Loss
#https://arxiv.org/abs/2102.02696
class ABL(nn.Module):
    def __init__(self, isdetach=True, max_N_ratio = 1/100, ignore_label = 255, label_smoothing=0.0, weight = None, max_clip_dist = 20.):
        super(ABL, self).__init__()
        self.ignore_label = ignore_label
        self.label_smoothing = label_smoothing
        self.isdetach=isdetach
        self.max_N_ratio = max_N_ratio

        self.weight_func = lambda w, max_distance=max_clip_dist: torch.clamp(w, max=max_distance) / max_distance

        self.dist_map_transform = transforms.Compose([
            lambda img: img.unsqueeze(0),
            lambda nd: nd.type(torch.int64),
            partial(class2one_hot, C=1),
            itemgetter(0),
            lambda t: t.cpu().numpy(),
            one_hot2dist,
            lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])

        if label_smoothing == 0:
            self.criterion = nn.CrossEntropyLoss(
                weight=weight,
                ignore_index=ignore_label,
                reduction='none'
            )
        else:
            self.criterion = LSSCE(
                reduction='none',
                ignore_index=ignore_label,
                lb_smooth = label_smoothing
            )

    def logits2boundary(self, logit):
        eps = 1e-5
        _, _, h, w = logit.shape
        max_N = (h*w) * self.max_N_ratio
        kl_ud = kl_div(logit[:, :, 1:, :], logit[:, :, :-1, :]).sum(1, keepdim=True)
        kl_lr = kl_div(logit[:, :, :, 1:], logit[:, :, :, :-1]).sum(1, keepdim=True)
        kl_ud = torch.nn.functional.pad(
            kl_ud, [0, 0, 0, 1, 0, 0, 0, 0], mode='constant', value=0)
        kl_lr = torch.nn.functional.pad(
            kl_lr, [0, 1, 0, 0, 0, 0, 0, 0], mode='constant', value=0)
        kl_combine = kl_lr+kl_ud
        while True: # avoid the case that full image is the same color
            kl_combine_bin = (kl_combine > eps).to(torch.float)
            if kl_combine_bin.sum() > max_N:
                eps *=1.2
            else:
                break
        #dilate
        dilate_weight = torch.ones((1,1,3,3)).cuda()
        edge2 = torch.nn.functional.conv2d(kl_combine_bin, dilate_weight, stride=1, padding=1)
        edge2 = edge2.squeeze(1)  # NCHW->NHW
        kl_combine_bin = (edge2 > 0)
        return kl_combine_bin

    def gt2boundary(self, gt, ignore_label=-1):  # gt NHW
        gt_ud = gt[:,1:,:]-gt[:,:-1,:]  # NHW
        gt_lr = gt[:,:,1:]-gt[:,:,:-1]
        gt_ud = torch.nn.functional.pad(gt_ud, [0,0,0,1,0,0], mode='constant', value=0) != 0 
        gt_lr = torch.nn.functional.pad(gt_lr, [0,1,0,0,0,0], mode='constant', value=0) != 0
        gt_combine = gt_lr+gt_ud
        del gt_lr
        del gt_ud
        
        # set 'ignore area' to all boundary
        gt_combine += (gt==ignore_label)
        
        return gt_combine > 0

    def get_direction_gt_predkl(self, pred_dist_map, pred_bound, logits):
        # NHW,NHW,NCHW
        eps = 1e-5
        # bound = torch.where(pred_bound)  # 3k
        bound = torch.nonzero(pred_bound*1)
        n,x,y = bound.T
        max_dis = 1e5

        logits = logits.permute(0,2,3,1) # NHWC

        pred_dist_map_d = torch.nn.functional.pad(pred_dist_map,(1,1,1,1,0,0),mode='constant', value=max_dis) # NH+2W+2

        logits_d = torch.nn.functional.pad(logits,(0,0,1,1,1,1,0,0),mode='constant') # N(H+2)(W+2)C
        logits_d[:,0,:,:] = logits_d[:,1,:,:] # N(H+2)(W+2)C
        logits_d[:,-1,:,:] = logits_d[:,-2,:,:] # N(H+2)(W+2)C
        logits_d[:,:,0,:] = logits_d[:,:,1,:] # N(H+2)(W+2)C
        logits_d[:,:,-1,:] = logits_d[:,:,-2,:] # N(H+2)(W+2)C
        
        """
        | 4| 0| 5|
        | 2| 8| 3|
        | 6| 1| 7|
        """
        x_range = [1, -1,  0, 0, -1,  1, -1,  1, 0]
        y_range = [0,  0, -1, 1,  1,  1, -1, -1, 0]
        dist_maps = torch.zeros((0,len(x))).cuda() # 8k
        kl_maps = torch.zeros((0,len(x))).cuda() # 8k

        kl_center = logits[(n,x,y)] # KC

        for dx, dy in zip(x_range, y_range):
            dist_now = pred_dist_map_d[(n,x+dx+1,y+dy+1)]
            dist_maps = torch.cat((dist_maps,dist_now.unsqueeze(0)),0)

            if dx != 0 or dy != 0:
                logits_now = logits_d[(n,x+dx+1,y+dy+1)]
                # kl_map_now = torch.kl_div((kl_center+eps).log(), logits_now+eps).sum(2)  # 8KC->8K
                if self.isdetach:
                    logits_now = logits_now.detach()
                kl_map_now = kl_div(kl_center, logits_now)
                
                kl_map_now = kl_map_now.sum(1)  # KC->K
                kl_maps = torch.cat((kl_maps,kl_map_now.unsqueeze(0)),0)
                torch.clamp(kl_maps, min=0.0, max=20.0)

        # direction_gt shound be Nk  (8k->K)
        direction_gt = torch.argmin(dist_maps, dim=0)
        # weight_ce = pred_dist_map[bound]
        weight_ce = pred_dist_map[(n,x,y)]
        # print(weight_ce)

        # delete if min is 8 (local position)
        direction_gt_idx = [direction_gt!=8]
        direction_gt = direction_gt[direction_gt_idx]


        kl_maps = torch.transpose(kl_maps,0,1)
        direction_pred = kl_maps[direction_gt_idx]
        weight_ce = weight_ce[direction_gt_idx]

        return direction_gt, direction_pred, weight_ce

    def get_dist_maps(self, target):
        target_detach = target.clone().detach()
        dist_maps = torch.cat([self.dist_map_transform(target_detach[i]) for i in range(target_detach.shape[0])])
        out = -dist_maps
        out = torch.where(out>0, out, torch.zeros_like(out))
        
        return out

    def forward(self, logits, target):
        eps = 1e-10
        ph, pw = logits.size(2), logits.size(3)
        h, w = target.size(1), target.size(2)

        if ph != h or pw != w:
            logits = F.interpolate(input=logits, size=(
                h, w), mode='bilinear', align_corners=True)

        gt_boundary = self.gt2boundary(target, ignore_label=self.ignore_label)

        dist_maps = self.get_dist_maps(gt_boundary).cuda() # <-- it will slow down the training, you can put it to dataloader.

        pred_boundary = self.logits2boundary(logits)
        if pred_boundary.sum() < 1: # avoid nan
            return None # you should check in the outside. if None, skip this loss.
        
        direction_gt, direction_pred, weight_ce = self.get_direction_gt_predkl(dist_maps, pred_boundary, logits) # NHW,NHW,NCHW

        # direction_pred [K,8], direction_gt [K]
        loss = self.criterion(direction_pred, direction_gt) # careful
        
        weight_ce = self.weight_func(weight_ce)
        loss = (loss * weight_ce).mean()  # add distance weight

        return loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GALoss(nn.Module):
    """
    The Class of GALoss
    """
    def __init__(self):
        super(GALoss, self).__init__()
        self._smooth = 1

    def _dice_loss(self, predict, target):
        """
        Compute the dice loss of the prediction decision map and ground-truth label
        :param predict: tensor, the prediction decision map
        :param target: tensor, ground-truth label
        :return:
        """
        # predict = torch.functional.F.softmax(predict, dim=1)
        n = predict.shape[0]
        target = target.float()
        # predict = predict.view(-1)
        # target = target.view(-1)
        intersect = torch.sum(predict * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(predict * predict)
        loss = (2 * intersect + self._smooth) / (z_sum + y_sum + self._smooth)
        loss = 1 - loss
        loss = torch.cosh(loss)
        loss = torch.log(loss)
        loss = torch.mean(loss)
        return loss

    def _qg_soft(self, img1, img2, fuse, k):
        """
        Compute the Qg for the given two image and the fused image.
        The calculation of Qg is modified to the python version based on the
        matlab version from https://github.com/zhengliu6699/imageFusionMetrics
        :param img1: tensor, input image A
        :param img2: tensor, input image B
        :param fuse: tensor, fused image
        :param k: softening factor
        :return:
        """
        # 1) get the map
        img1_gray = img1
        img2_gray = img2
        buf = 0.000001
        flt1 = torch.FloatTensor(np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1], ])).reshape((1, 1, 3, 3)).cuda(img1.device)
        flt2 = torch.FloatTensor(np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1], ])).reshape((1, 1, 3, 3)).cuda(img1.device)
        fuseX = F.conv2d(fuse, flt1, padding=1) + buf
        fuseY = F.conv2d(fuse, flt2, padding=1)
        fuseG = torch.sqrt(torch.mul(fuseX, fuseX) + torch.mul(fuseY, fuseY))
        buffer = (fuseX == 0)
        buffer = buffer.float()
        buffer = buffer * buf
        fuseX = fuseX + buffer
        fuseA = torch.atan(torch.div(fuseY, fuseX))

        img1X = F.conv2d(img1_gray, flt1, padding=1)
        img1Y = F.conv2d(img1_gray, flt2, padding=1)
        img1G = torch.sqrt(torch.mul(img1X, img1X) + torch.mul(img1Y, img1Y))
        buffer = (img1X == 0)
        buffer = buffer.float()
        buffer = buffer * buf
        img1X = img1X + buffer
        img1A = torch.atan(torch.div(img1Y, img1X))

        img2X = F.conv2d(img2_gray, flt1, padding=1)
        img2Y = F.conv2d(img2_gray, flt2, padding=1)
        img2G = torch.sqrt(torch.mul(img2X, img2X) + torch.mul(img2Y, img2Y))
        buffer = (img2X == 0)
        buffer = buffer.float()
        buffer = buffer * buf
        img2X = img2X + buffer
        img2A = torch.atan(torch.div(img2Y, img2X))

        # 2) edge preservation estimation

        buffer = (img1G == 0)
        buffer = buffer.float()
        buffer = buffer * buf
        img1G = img1G + buffer
        buffer1 = torch.div(fuseG, img1G)

        buffer = (fuseG == 0)
        buffer = buffer.float()
        buffer = buffer * buf
        fuseG = fuseG + buffer
        buffer2 = torch.torch.div(img1G, fuseG)

        bimap = torch.sigmoid(-k * (img1G - fuseG))
        bimap_1 = torch.sigmoid(k * (img1A - fuseA))
        Gaf = torch.mul(bimap, buffer2)+torch.mul((1 - bimap), buffer1)
        Aaf = torch.abs(torch.abs(img1A - fuseA) - np.pi/2)*2/np.pi

        # -------------------
        buffer = (img2G == 0)
        buffer = buffer.float()
        buffer = buffer*buf
        img2G = img2G+buffer
        buffer1 = torch.div(fuseG, img2G)

        buffer = (fuseG == 0)
        buffer = buffer.float()
        buffer = buffer*buf
        fuseG = fuseG+buffer
        buffer2 = torch.div(img2G, fuseG)

        # bimap = torch.sigmoid(-k * (img2G-fuseG))
        bimap = torch.sigmoid(-k * (img2G-fuseG))
        bimap_2 = torch.sigmoid(k * (img2A-fuseA))
        Gbf = torch.mul(bimap, buffer2)+torch.mul((1-bimap), buffer1)
        Abf = torch.abs(torch.abs(img2A-fuseA) - np.pi/2) * 2 / np.pi

        # some parameter
        gama1 = 1
        gama2 = 1
        k1 = -10
        k2 = -20
        delta1 = 0.5
        delta2 = 0.75

        Qg_AF = torch.div(gama1, (1 + torch.exp(k1 * (Gaf - delta1))))
        Qalpha_AF = torch.div(gama2, (1+torch.exp(k2 * (Aaf - delta2))))
        Qaf = torch.mul(Qg_AF, Qalpha_AF)

        Qg_BF = torch.div(gama1, (1 + torch.exp(k1 * (Gbf - delta1))))
        Qalpha_BF = torch.div(gama2, (1 + torch.exp(k2 * (Abf - delta2))))
        Qbf = torch.mul(Qg_BF, Qalpha_BF)

        # 3) compute the weighting matrix
        L = 1
        Wa = torch.pow(img1G, L)
        Wb = torch.pow(img2G, L)
        res = torch.mean(torch.div(torch.mul(Qaf, Wa) + torch.mul(Qbf, Wb), (Wa + Wb)))

        return res
    def forward(self, mask, gt_mask, k=10e4):
        """
        Compute the GALoss
        :param img1: tensor, input image A
        :param img2: tensor, input image B
        :param mask: tensor, the prediction decision map without bounary guider filter
        :param mask_BGF: tensor, the prediction decision map with bounary guider filter
        :param gt_mask: tensor, the ground-truth decision map
        :param k: the softening factor of loss_qg
        :return:
        """

        loss_dice = self._dice_loss(mask, gt_mask)

        return loss_dice
    # def forward(self, img1, img2, mask, mask_BGF, gt_mask, k=10e4):
    #     """
    #     Compute the GALoss
    #     :param img1: tensor, input image A
    #     :param img2: tensor, input image B
    #     :param mask: tensor, the prediction decision map without bounary guider filter
    #     :param mask_BGF: tensor, the prediction decision map with bounary guider filter
    #     :param gt_mask: tensor, the ground-truth decision map
    #     :param k: the softening factor of loss_qg
    #     :return:
    #     """
    #     fused = torch.mul(mask_BGF, img1) + torch.mul((1 - mask_BGF), img2)
    #     loss_qg = 1 - self._qg_soft(img1, img2, fused, k)
    #     loss_dice = self._dice_loss(mask, gt_mask)

    #     return loss_dice + loss_qg, loss_dice, loss_qg
# from torch.backends import cudnn
# import os
# import random
# cudnn.benchmark = False
# cudnn.deterministic = True

# seed = 0
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

# random.seed(seed)
# np.random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)

# n,c,h,w = 1,2,100,100
# gt = torch.zeros((n,c,h,w)).cuda()
# gt[0,:,5] = 1
# gt[0,:,50] = 1
# logits = torch.randn((n,c,h,w)).cuda()

# abl = ABL()
# print(abl(logits, gt))