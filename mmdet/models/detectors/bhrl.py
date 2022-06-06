import torch
import torch.nn as nn

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import DETECTORS, build_roi_extractor, build_head, HEADS
from .two_stage import TwoStageDetector
from ..plugins.match_module import MatchModule
from ..plugins.generate_ref_roi_feats import generate_ref_roi_feats
from mmcv.cnn import xavier_init
import mmcv
import numpy as np
from mmcv.image import imread, imwrite
import cv2
from mmcv.visualization.color import color_val
from random import  choice
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.models.roi_heads.test_mixins import BBoxTestMixin, MaskTestMixin

@DETECTORS.register_module()
class BHRL(TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(BHRL, self).__init__(backbone=backbone,
                                                    neck=neck,
                                                    rpn_head=rpn_head,
                                                    roi_head=roi_head,
                                                    train_cfg=train_cfg,
                                                    test_cfg=test_cfg,
                                                    pretrained=pretrained,
                                                    init_cfg=init_cfg)

        self.matching_block = MatchModule(512, 384)
    
    def matching(self, img_feat, rf_feat):
        out = []
        for i in range(len(rf_feat)):
            out.append(self.matching_block(img_feat[i], rf_feat[i]))
        return out

    def extract_feat(self, img):
        img_feat = img[0]
        rf_feat = img[1]
        rf_bbox = img[2]
        img_feat = self.backbone(img_feat)
        rf_feat = self.backbone(rf_feat)
        if self.with_neck:
            img_feat = self.neck(img_feat)
            rf_feat = self.neck(rf_feat)

        img_feat_metric = self.matching(img_feat, rf_feat)

        ref_roi_feats = generate_ref_roi_feats(rf_feat, rf_bbox)
        return tuple(img_feat_metric), tuple(img_feat), ref_roi_feats

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x, img_feat, ref_roi_feats = self.extract_feat(img)

        losses = dict()

        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_feat, ref_roi_feats, 
                                                 img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)

        losses.update(roi_losses)

        return losses
        
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x, img_feat, ref_roi_feats = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, img_feat, ref_roi_feats, proposal_list, img_metas, rescale=rescale)
    
    