#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import build_activation_layer, build_norm_layer

EPSILON = 1e-12


class AttentionAwareModule(nn.Module):
    """Attention-Aware Module with Bilinear Attention Pooling"""

    def __init__(
        self,
        in_channels,
        out_channels,
        att_channels=32,
        pool="GAP",
        norm_cfg=dict(type="BN1d", requires_grad=True),
        act_cfg=dict(type="ReLU"),
    ):
        super().__init__()

        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.att = nn.Conv2d(out_channels, att_channels, kernel_size=1)

        assert pool in ["GAP", "GMP"]
        if pool == "GAP":
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

        self.norm = build_norm_layer(norm_cfg, out_channels * att_channels)[1]
        self.act = build_activation_layer(act_cfg)

    def init_weights(self):
        # conv
        nn.init.kaiming_normal_(self.reduce.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.att.weight, mode="fan_in")

        # bn
        nn.init.normal_(self.norm.weight, mean=1.0, std=0.02)
        nn.init.constant_(self.norm.bias, 0.0)

    def forward(self, x):
        x = self.reduce(x)
        attentions = self.att(x)
        B, C, H, W = x.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature = (
                torch.einsum("imjk,injk->imn", (attentions, x)) / float(H * W)
            ).view(B, -1)
        else:
            feature = []
            for i in range(M):
                AiF = self.pool(x * attentions[:, i : i + 1, ...]).view(B, -1)
                feature.append(AiF)
            feature = torch.cat(feature, dim=1)

        # sign-sqrt
        output = torch.sign(feature) * torch.sqrt(torch.abs(feature) + EPSILON)

        # l2 normalization along dimension M and C
        # feature = F.normalize(feature, dim=-1)

        # normalize output
        output = self.norm(output)
        output = self.act(output)
        return output


class Pruning(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_cfg=dict(type="BN2d", requires_grad=True),
        act_cfg=dict(type="ReLU"),
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.norm = build_norm_layer(cfg=norm_cfg, num_features=out_channels)[1]
        self.act = build_activation_layer(act_cfg)

    def init_weights(self):
        # conv
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in")

        # bn
        nn.init.normal_(self.norm.weight, mean=1.0, std=0.02)
        nn.init.constant_(self.norm.bias, 0.0)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
    ):
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)

    def init_weights(self):
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_out")
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        return self.fc(x)


class PartClassifier(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        num_parts,
    ):
        super().__init__()

        assert num_parts > 1
        self.num_parts = num_parts

        fcs = []
        for _ in range(num_parts):
            fcs.append(Classifier(in_channels, num_classes))
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
