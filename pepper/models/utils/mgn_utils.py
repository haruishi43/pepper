#!/usr/bin/env python3

import torch.nn as nn

from mmcv.cnn import build_activation_layer, build_norm_layer


class Pruning(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_cfg=dict(type="BN2d", requires_grad=True),
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.norm = build_norm_layer(cfg=norm_cfg, num_features=out_channels)[1]

    def init_weights(self):
        # conv
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in")

        # bn
        nn.init.normal_(self.norm.weight, mean=1.0, std=0.02)
        nn.init.constant_(self.norm.bias, 0.0)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        act_cfg=dict(type="ReLU"),
    ):
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)
        self.act = build_activation_layer(act_cfg)

    def init_weights(self):
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_out")
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        x = self.act(x)  # NOTE: added activation
        return self.fc(x)


class PartClassifier(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        num_parts,
        act_cfg=dict(type="ReLU"),
    ):
        super().__init__()

        assert num_parts > 1
        self.num_parts = num_parts

        fcs = []
        for _ in range(num_parts):
            fcs.append(Classifier(in_channels, num_classes, act_cfg=act_cfg))
        self.fcs = nn.ModuleList(fcs)

    def init_weights(self):
        for fc in self.fcs:
            fc.init_weights()

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        assert len(x) == self.num_parts

        outs = []
        for xi, fc in zip(x, self.fcs):
            xi = fc(xi)
            outs.append(xi)

        return outs
