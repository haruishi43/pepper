#!/usr/bin/env python3

import torch.nn as nn
import torch.nn.functional as F

from mmcls.models.builder import BACKBONES
from mmcls.models.backbones.base_backbone import BaseBackbone


@BACKBONES.register_module()
class LeNetPlusPlus(BaseBackbone):
    """LeNet++ as described in the Center Loss paper"""

    def __init__(
        self,
        num_classes=-1,
        init_cfg=None,
    ):
        super(LeNetPlusPlus, self).__init__(init_cfg=init_cfg)

        self.num_classes = num_classes

        self.conv1_1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.prelu1_2 = nn.PReLU()

        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.prelu2_2 = nn.PReLU()

        self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.prelu3_2 = nn.PReLU()

        if self.num_classes > 0:
            self.classifier = nn.Linear(128 * 3 * 3, num_classes)

        # out channel = 128 * 3 * 3 = 1152
        # self.init_weights()

    def init_weights(self):
        ...

    def forward(self, x):

        if isinstance(x, (tuple, list)):
            x = x[-1]

        x = self.prelu1_1(self.bn1_1(self.conv1_1(x)))
        x = self.prelu1_2(self.bn1_2(self.conv1_2(x)))
        x = F.max_pool2d(x, 2)

        x = self.prelu2_1(self.bn2_1(self.conv2_1(x)))
        x = self.prelu2_2(self.bn2_2(self.conv2_2(x)))
        x = F.max_pool2d(x, 2)

        x = self.prelu3_1(self.bn3_1(self.conv3_1(x)))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)

        # TODO: probably should move it to the neck
        x = x.view(-1, 128 * 3 * 3)  # [bs, 1152]

        if self.num_classes > 0:
            x = self.classifier(x)

        return x
