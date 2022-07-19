#!/usr/bin/env python3

import torch
import torch.nn as nn

from mmcv.runner import auto_fp16

from .basic_head import BasicHead
from ..builder import HEADS


@HEADS.register_module()
class BaselineHead(BasicHead):
    """Bag-of-Trick Baseline Head

    No batch norm layers before classifier
    """

    def _init_layers(self):
        if self.loss_cls:

            # self.bn = nn.BatchNorm1d(self.in_channels)
            # self.bn.bias.requires_grad_(False)

            self.classifier = nn.Linear(
                self.in_channels, self.num_classes, bias=False
            )

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

        return x

    @auto_fp16()
    def forward_train(self, x):
        """Model forward."""
        if self.loss_cls:
            # feat = self.bn(x)
            cls_score = self.classifier(x)
            return (x, cls_score)
        return (x,)
