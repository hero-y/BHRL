import torch.nn as nn
from mmcv.cnn import normal_init, xavier_init

from mmdet.models.backbones.resnet import Bottleneck
from mmdet.models.backbones.resnext import Bottleneck as resnext_Bottleneck
from ...builder import HEADS
from mmcv.cnn import ConvModule
from .bbox_head import BBoxHead
import numpy as np
import torch
from mmdet.models.plugins import IHR
from ...builder import build_loss
import torch
import torch.nn.functional as F
from ...losses import accuracy
# from mmdet.core import (auto_fp16, bbox_target, delta2bbox, force_fp32,
#                         multiclass_nms)
from mmcv.runner import auto_fp16, force_fp32
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule

@HEADS.register_module()
class BHRLConvFCBBoxHead(BBoxHead):

    def __init__(self,
                 use_shared_fc = False,
                 num_convs=0,
                 num_fcs=0,
                 conv_out_channels=1024,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 ihr = None,
                 **kwargs):
        kwargs.setdefault('with_avg_pool', True)
        super(BHRLConvFCBBoxHead, self).__init__(**kwargs)
        assert self.with_avg_pool
        self.use_shared_fc = use_shared_fc
        assert num_fcs > 0
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.fc_branch = self._add_fc_branch()

        out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
        self.fc_reg = nn.Linear(self.conv_out_channels, out_dim_reg)
        
        self.fc_cls = nn.Linear(self.fc_out_channels, self.num_classes + 1)
       
        self.relu = nn.ReLU(inplace=True)

        self.metric_module = IHR(ihr)

    def _add_conv_branch(self):
        """Add the fc branch which consists of a sequential of conv layers"""
        branch_convs = nn.ModuleList()
        for i in range(self.num_convs):
            branch_convs.append(
                Bottleneck(
                    inplanes=self.conv_out_channels,
                    planes=self.conv_out_channels // 4,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        return branch_convs

    def _add_fc_branch(self):
        """Add the fc branch which consists of a sequential of fc layers"""
        branch_fcs = nn.ModuleList()
        for i in range(self.num_fcs):
            fc_in_channels = (
                self.in_channels *
                self.roi_feat_area if i == 0 else self.fc_out_channels)
            branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
        return branch_fcs

    def init_weights(self):
        normal_init(self.fc_cls, std=0.01)
        normal_init(self.fc_reg, std=0.001)

        for m in self.fc_branch.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')

    def forward(self, target, rois=None, query=None):
 
        inds = rois[:,0]

        relation = self.metric_module(target,query[inds.long()])

        if relation is not None:
            relation_fc = relation.view(relation.size(0), -1)
            for fc in self.fc_branch:
                relation_fc = self.relu(fc(relation_fc))
            cls_score = self.fc_cls(relation_fc)
        else:
            cls_score = None
    
        bbox_pred = self.fc_reg(relation_fc)

        return cls_score, bbox_pred
        
