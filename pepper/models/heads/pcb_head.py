#!/usr/bin/env python3

import torch
import torch.nn as nn

from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.runner import auto_fp16, force_fp32

from .basic_head import BasicHead
from .utils import weights_init_classifier, weights_init_kaiming
from ..builder import HEADS


class PartFeatureBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        norm_cfg=dict(type="BN1d", requires_grad=True),
        act_cfg=dict(type="ReLU"),
    ):
        super().__init__()
        self.reduce = nn.Linear(
            in_channels,
            mid_channels,
            bias=False,
        )
        _, self.bn = build_norm_layer(norm_cfg, mid_channels)
        self.act = build_activation_layer(act_cfg)

    def init_weights(self):
        self.reduce.apply(weights_init_kaiming)
        self.bn.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.reduce(x)
        x = self.bn(x)

        if self.training:
            return self.act(x)
        else:
            return x


@HEADS.register_module()
class PCBHead(BasicHead):
    def __init__(
        self,
        num_parts,
        mid_channels=256,
        **kwargs,
    ):
        assert num_parts > 0
        self.num_parts = num_parts
        self.mid_channels = mid_channels
        super().__init__(**kwargs)

    def _init_layers(self):

        blocks = []
        fcs = []
        for _ in range(self.num_parts):
            blocks.append(PartFeatureBlock(self.in_channels, self.mid_channels))
            fcs.append(
                nn.Linear(self.mid_channels, self.num_classes)  # , bias=False)
            )

        self.blocks = nn.ModuleList(blocks)
        self.fcs = nn.ModuleList(fcs)

    def init_weights(self):
        for b in self.blocks:
            b.init_weights()
        for f in self.fcs:
            f.apply(weights_init_classifier)

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

        outs = []
        for i, block in enumerate(self.blocks):
            x_i = x[:, :, i]
            x_i = block(x_i)
            outs.append(x_i)

        if self.training:
            return outs
        else:
            # instead of cat, we can also take the mean?
            return torch.cat(outs, dim=-1)

    @auto_fp16()
    def forward_train(self, x):

        if self.loss_cls:
            cls_scores = []
            for x_i, fc in zip(x, self.fcs):
                x_i = fc(x_i)
                cls_scores.append(x_i)
            return (x, cls_scores)
        return (x,)

    @force_fp32(apply_to=("feats", "cls_score"))
    def loss(self, gt_label, feats, cls_score=None):
        """Compute losses."""

        losses = dict()

        # Classification (softmax) losses
        if self.loss_cls:
            assert cls_score is not None
            assert len(cls_score) == self.num_parts

            if not isinstance(self.loss_cls, nn.ModuleList):
                loss_cls = [self.loss_cls]
            else:
                loss_cls = self.loss_cls

            for i, cs in enumerate(cls_score):
                for lc in loss_cls:
                    if lc.loss_name not in losses:
                        losses[lc.loss_name] = lc(
                            cls_score=cs,
                            label=gt_label,
                        )
                    else:
                        losses[lc.loss_name] += lc(
                            cls_score=cs,
                            label=gt_label,
                        )

                # compute accuracy
                acc = self.accuracy(cs, gt_label)
                assert len(acc) == len(self.topk)
                if "accuracy" not in losses:

                    losses["accuracy"] = {
                        f"part{i}_top-{k}": a for k, a in zip(self.topk, acc)
                    }
                else:
                    accuracy = {
                        f"part{i}_top-{k}": a for k, a in zip(self.topk, acc)
                    }
                    losses["accuracy"].update(accuracy)

        # Metric (distance) losses
        if self.loss_pairwise:
            assert len(feats) == self.num_parts

            if not isinstance(self.loss_pairwise, nn.ModuleList):
                loss_pairwise = [self.loss_pairwise]
            else:
                loss_pairwise = self.loss_pairwise

            for f in feats:
                for lp in loss_pairwise:
                    if lp.loss_name not in losses:
                        losses[lp.loss_name] = lp(
                            inputs=f,
                            targets=gt_label,
                        )
                    else:
                        losses[lp.loss_name] += lp(
                            inputs=f,
                            targets=gt_label,
                        )

        return losses
