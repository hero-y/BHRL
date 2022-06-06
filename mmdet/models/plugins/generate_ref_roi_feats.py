
from torchvision.ops import roi_align
import torch.nn as nn
import torch
import torch.nn.functional as F
import mmcv
import numpy as np
import cv2

def forward_fuse(feats):
    feats = list(feats)
    feats[0] = feats[0].unsqueeze(1)
    for i in range(1, len(feats)):
        feats[i] = F.interpolate(feats[i], scale_factor=2 ** i, mode='nearest')
        feats[i] = feats[i].unsqueeze(1)
    feats = torch.cat(feats, dim=1)
    feats = feats.mean(dim=1)
    return feats


def generate_ref_roi_feats(rf_feat, bbox):
    ref_fuse_feats = forward_fuse(rf_feat)
    roi_feats = []
    for j in range(bbox.shape[0]):
        roi_feat = roi_align(ref_fuse_feats[j].unsqueeze(0), [bbox[j] / 4], [7, 7])
        roi_feats.append(roi_feat)
    roi_feats = torch.cat(roi_feats, dim=0)
    return roi_feats

