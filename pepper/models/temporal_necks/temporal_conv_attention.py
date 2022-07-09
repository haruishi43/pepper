#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseTemporalLayer
from ..builder import TEMPORAL


@TEMPORAL.register_module()
class TemporalConvAttention(BaseTemporalLayer):
    """Temporal Conv Attention for ResNet output"""

    attentions = ("softmax", "sigmoid")

    def __init__(
        self,
        in_channels,
        mid_dim=256,
        attention_mode="softmax",
        kernel_size=(7, 4),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.last_dim = in_channels
        self.mid_dim = mid_dim

        assert attention_mode in self.attentions
        self.att_gen = attention_mode

        # input size [3, 256, 128] -> [2048, 8, 4]
        # input size [3, 224, 112] -> [2048, 7, 4]
        self.att_conv = nn.Conv2d(
            in_channels,
            mid_dim,
            kernel_size=kernel_size,
        )
        self.att_t_conv = nn.Conv2d(self.mid_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _forward(self, x):
        b, s, f_dim, h, w = x.shape
        x = x.view(b * s, f_dim, h, w)

        a = self.relu(self.att_conv(x))
        a = a.view(b, s, -1)
        a = a.permute(0, 2, 1).unsqueeze(-1)
        a = self.relu(self.att_t_conv(a))
        a = a.view(b, s)

        x = F.avg_pool2d(x, x.shape[2:])

        if self.att_gen == "softmax":
            # NOTE: gives harsh values that are either too small or too large at first
            # then settles to none harsh values
            a = torch.softmax(a, dim=1)
        elif self.att_gen == "sigmoid":
            # NOTE: sigmoid gives good values that aren't so harsh
            a = torch.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)
        else:
            raise KeyError(
                f"Unsupported attention generation function: {self.att_gen}"
            )

        x = x.view(b, s, -1)
        a = a.unsqueeze(-1)
        a = a.expand(b, s, self.last_dim)
        x = torch.mul(x, a)
        x = torch.sum(x, 1)
        x = x.view(b, self.last_dim)

        return x
