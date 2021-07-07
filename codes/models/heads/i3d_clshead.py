import torch.nn as nn

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module
class I3DClsHead(BaseHead):

    def __init__(self,
                 spatial_type='avg',
                 spatial_size=7,
                 temporal_size=4,
                 consensus_cfg=dict(type='avg', dim=1),
                 dropout_ratio=0.5,
                 in_channels=2048,
                 num_classes=400,
                 init_std=0.01,
                 fcn_testing=False):
        super(I3DClsHead, self).__init__(spatial_size, dropout_ratio,
                                         in_channels, num_classes, init_std)
        self.spatial_type = spatial_type
        self.consensus_type = consensus_cfg['type']

        self.temporal_size = temporal_size
        assert not (self.spatial_size == -1) ^ (self.temporal_size == -1)

        if self.temporal_size == -1 and self.spatial_size == -1:
            self.pool_size = (1, 1, 1)
            if self.spatial_type == 'avg':
                self.Logits = nn.AdaptiveAvgPool3d(self.pool_size)
            if self.spatial_type == 'max':
                self.Logits = nn.AdaptiveMaxPool3d(self.pool_size)
        else:
            self.pool_size = (self.temporal_size, ) + self.spatial_size
            if self.spatial_type == 'avg':
                self.Logits = nn.AvgPool3d(
                    self.pool_size, stride=1, padding=0)
            if self.spatial_type == 'max':
                self.Logits = nn.MaxPool3d(
                    self.pool_size, stride=1, padding=0)

        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        self.fcn_testing = fcn_testing
        self.new_cls = None

    def forward(self, x, num_clips, num_crops):
        if not self.fcn_testing:
            x = self.Logits(x)
            if self.dropout is not None:
                x = self.dropout(x)
            x = x.view(x.shape[0], -1)
            cls_score = self.fc_cls(x)
            cls_score = cls_score.reshape((-1, num_clips*num_crops) + cls_score.shape[1:])
            cls_score = cls_score.mean(dim=1)
        else:
            if self.new_cls is None:
                self.new_cls = nn.Conv3d(
                    self.in_channels,
                    self.num_classes,
                    1, 1, 0).cuda()
                self.new_cls.load_state_dict(
                    {'weight': self.fc_cls.weight.unsqueeze(-1).unsqueeze(
                        -1).unsqueeze(-1),
                     'bias': self.fc_cls.bias})
            class_map = self.new_cls(x)
            cls_score = class_map.mean([2, 3, 4])
            cls_score = cls_score.reshape((-1, num_clips*num_crops) + cls_score.shape[1:])
            cls_score = cls_score.mean(dim=1)

        return cls_score

    def init_weights(self):
        nn.init.normal_(self.fc_cls.weight, 0, self.init_std)
        nn.init.constant_(self.fc_cls.bias, 0)
