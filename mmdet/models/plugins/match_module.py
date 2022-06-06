import torch
import torch.nn as nn

from mmcv.cnn import xavier_init
import mmcv
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.ops import roi_align

class MatchModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):
        super(MatchModule, self).__init__()

        self.metric_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.init_weights()

    def init_weights(self, std=0.01):
        xavier_init(self.metric_conv, distribution='uniform')

    def forward(self, x, ref):
        ref_avg = self.avg(ref)
        delta = (x - ref_avg).abs()
        feat = torch.cat((x, delta), dim=1)
        output = self.metric_conv(feat)
        return output
