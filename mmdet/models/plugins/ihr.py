import torch
import torch.nn as nn

from mmcv.cnn import xavier_init
import mmcv
import torch.nn.functional as F
import numpy as np
import cv2

class IHR(nn.Module):

    def __init__(self,
                 metric_module):
        super(IHR, self).__init__()

        in_channel = metric_module['metric_module_in_channel']
        out_channel = metric_module['metric_module_out_channel']

        self.key_t = nn.Conv2d(in_channel, in_channel//8, kernel_size=(3,3), padding=(1,1), stride=1)
        self.key_q = nn.Conv2d(in_channel, in_channel//8, kernel_size=(3,3), padding=(1,1), stride=1)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.metric = nn.Conv2d(in_channel*2, out_channel, 1, 1)
        self.conv1 = nn.Conv2d(in_channel, in_channel//2, 1, 1)

        xavier_init(self.key_t, distribution='uniform')
        xavier_init(self.key_q, distribution='uniform')
        xavier_init(self.metric, distribution='uniform')
        xavier_init(self.conv1, distribution='uniform')
  

    def forward(self, x, ref):

        bs = ref.shape[0]
        ref_h, ref_w = ref.shape[2:]
        x_h, x_w = x.shape[2:]

        contrastive_feat = (self.avg(ref) - x).abs()
        contrastive_feat = self.conv1(contrastive_feat)

        salient_feat = F.conv2d(x.reshape(1,-1,x.shape[2],x.shape[3]), self.avg(ref).reshape(1,-1,self.avg(ref).shape[2],self.avg(ref).shape[3]).permute(1,0,2,3), groups=x.shape[0]*x.shape[1])
        salient_feat = salient_feat.reshape(x.shape[0],x.shape[1],x.shape[2],x.shape[3])
        salient_feat = self.conv1(salient_feat)

        key_q=self.key_q(ref)
        key_t=self.key_t(x)
        p = torch.matmul(key_t.view(bs,32,-1).permute(0,2,1),key_q.view(bs,32,-1))
        p = F.softmax(p,dim=1)
        val_t_out = torch.matmul(ref.view(bs,256,-1),p).view(bs,256,ref_h,ref_w) 
        attention_feat = (val_t_out - x).abs()
        attention_feat = self.conv1(attention_feat)

        contrastive_and_salient_feat= self.conv1(torch.cat([salient_feat,contrastive_feat],1))
        output = torch.cat([contrastive_and_salient_feat, attention_feat, x], 1)
        output = self.metric(output)

        return output
