#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseTemporalLayer
from ..builder import TEMPORAL


@TEMPORAL.register_module()
class TemporalAttention(BaseTemporalLayer):

    _attentions = ("softmax", "sigmoid")

    def __init__(
        self,
        in_channels,
        seq_len,
        attention_mode="softmax",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len

        assert attention_mode in self._attentions
        self.att_gen = attention_mode

        self.fc = nn.Linear(
            in_features=self.in_channels, out_features=1, bias=False
        )
        self.att_seq = nn.Linear(
            in_features=seq_len, out_features=1, bias=False
        )
        self.relu = nn.ReLU(inplace=True)

    def _forward(self, x, **kwargs):
        assert len(x.shape) == 3
        b, s, f_dim = x.shape
        assert f_dim == self.in_channels

        # Temporal Attention
        x1 = x.view(b * s, f_dim)
        x1 = self.relu(self.fc(x1))
        x1 = x1.view(b, s)

        # Feature Attention
        x2 = x.permute(0, 2, 1)  # (b, f_dim, s)
        x2 = x2.reshape(b * f_dim, s)
        x2 = self.relu(self.att_seq(x2))
        x2 = x2.view(b, f_dim)

        # multiply the temporal and feature attention
        att = torch.bmm(
            x1.unsqueeze(-1).expand(b, s, f_dim),
            x2.unsqueeze(-1).expand(b, f_dim, 1),
        ).squeeze(
            -1
        )  # (b, s)

        if self.att_gen == "softmax":
            att = torch.softmax(att, dim=1)
        elif self.att_gen == "sigmoid":
            att = torch.sigmoid(att)
            att = F.normalize(att, p=1, dim=1)

        # expand and multiply attention to input
        att = att.unsqueeze(-1).expand(b, s, f_dim)
        x = torch.mul(x, att)

        # take a sequential sum of the features
        x = torch.sum(x, 1)
        x = x.view(b, f_dim)  # reshape

        return x
