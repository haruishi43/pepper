#!/usr/bin/env python3

import torch
import torch.nn as nn

from mmcv.runner import auto_fp16

from .basic_head import BasicHead
from .utils import weights_init_classifier, weights_init_kaiming
from ..builder import HEADS


@HEADS.register_module()
class BoTHead(BasicHead):
    """Bag-of-Trick Head"""

    def _init_layers(self):
        self.bn = nn.BatchNorm1d(self.in_channels)
        self.bn.bias.requires_grad_(False)  # no shift (BoT)
        self.bn.apply(weights_init_kaiming)

        if self.loss_cls:
            self.classifier = nn.Linear(
                self.in_channels, self.num_classes, bias=False
            )
            self.classifier.apply(weights_init_classifier)

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

        # we use batch norm layer's output for inference
        bn_x = self.bn(x)

        return (x, bn_x)

    @auto_fp16()
    def forward_train(self, x):
        """Model forward."""

        assert len(x) == 2

        feats, feats_bn = x

        cls_score = self.classifier(feats_bn)
        return (feats, cls_score)
