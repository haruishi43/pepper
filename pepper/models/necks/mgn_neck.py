#!/usr/bin/env python3

import torch.nn as nn

from ..builder import NECKS


@NECKS.register_module()
class MGNPooling(nn.Module):
    def __init__(
        self,
        num2=2,
        num3=3,
    ):
        super().__init__()

        self.p1_global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.p2_global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.p3_global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.p2_part_pool = nn.AdaptiveAvgPool2d((num2, 1))
        self.p3_part_pool = nn.AdaptiveAvgPool2d((num3, 1))

        self.num2 = num2
        self.num3 = num3

    def init_weights(self):
        pass

    def forward(self, inputs):

        assert isinstance(inputs, (list, tuple))
        assert len(inputs) == 3

        p1 = inputs[0]
        p2 = inputs[1]
        p3 = inputs[2]

        p1_global = self.p1_global_pool(p1)  # .view(p1.size(0), -1)
        p2_global = self.p2_global_pool(p2)  # .view(p2.size(0), -1)
        p3_global = self.p3_global_pool(p3)  # .view(p3.size(0), -1)

        p2_parts = self.p2_part_pool(p2)  # .view(p2.size(0), self.num2, -1)
        p3_parts = self.p3_part_pool(p3)  # .view(p3.size(0), self.num3, -1)

        return (p1_global, p2_global, p3_global, p2_parts, p3_parts)
