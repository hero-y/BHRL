import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
import numpy as np

def rp_weight_generate(cls_score,label,alpha):
    cls_score_softmax = F.softmax(cls_score, dim=1)
    
    rp_weight = label.new_full((label.size(0), ), 1.0)
    rp_weight = rp_weight.float()

    proposals_num = label.size(0)

    inds_pos = torch.nonzero(label == 0).squeeze()
    inds_neg = torch.nonzero(label == 1).squeeze()
    
    inds_pos_num = inds_pos.numel()

    inds_fp = torch.nonzero(cls_score_softmax[inds_neg, 0] > cls_score_softmax[inds_neg, 1]).squeeze() 
    inds_fp_num = inds_fp.numel()
    
    inds_tn = torch.nonzero(cls_score_softmax[inds_neg, 0] < cls_score_softmax[inds_neg, 1]).squeeze() 
    inds_tn_num = inds_tn.numel() 
    
    if inds_pos_num > 0:
        rp_weight[inds_pos] = (proposals_num*alpha)/(inds_pos_num + inds_fp_num)  
    if inds_fp_num > 0:
        rp_weight[inds_neg[inds_fp]] = (proposals_num*alpha)/(inds_pos_num + inds_fp_num) 
    if inds_tn_num>0:
        rp_weight[inds_neg[inds_tn]] = (proposals_num*(1-alpha))/inds_tn_num
     
    return rp_weight


@LOSSES.register_module()
class RPLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=False,
                 reduction='mean',
                 loss_weight=1.0,
                 alpha=0.25):
        super(RPLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = F.cross_entropy(cls_score, label, reduction='none') 
        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()

        rp_weight = rp_weight_generate(cls_score,label,self.alpha)
        loss = loss * rp_weight
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
        return loss * self.loss_weight