#!/usr/bin/env python3

import torch
import torch.nn as nn

from mmcv.runner import auto_fp16

from mmcls.models.builder import HEADS

from .metric_head import MetricHead


@HEADS.register_module()
class VisualizeFeatureHead(MetricHead):
    def __init__(
        self,
        vis_dim=2,
        **kwargs,
    ):
        self.vis_dim = vis_dim
        super(VisualizeFeatureHead, self).__init__(**kwargs)

    def _init_layers(self):
        """Initialize layers"""

        self.bn = nn.BatchNorm1d(self.in_channels)
        self.bn.bias.requires_grad_(False)

        self.fc1 = nn.Linear(self.in_channels, self.vis_dim, bias=False)
        self.act = nn.PReLU()
        self.fc2 = nn.Linear(self.vis_dim, self.num_classes, bias=False)

    @auto_fp16()
    def pre_logits(self, x):
        # deals with tuple outputs from previous layers
        if isinstance(x, tuple):
            if len(x) > 1:
                # use the last one
                x = x[-1]
            else:
                x = x[0]

        assert isinstance(x, torch.Tensor)

        x = self.bn(x)
        x = self.act(self.fc1(x))
        cls_score = self.fc2(x)

        return dict(
            cls_score=cls_score,
            feats=x,
        )
