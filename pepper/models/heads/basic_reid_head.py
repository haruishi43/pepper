#!/usr/bin/env python3

import warnings

import torch
import torch.nn as nn

from mmcv.runner import auto_fp16, force_fp32

from .base_head import BaseHead
from ..builder import HEADS, build_loss
from ..losses import Accuracy


@HEADS.register_module()
class BasicReIDHead(BaseHead):
    """Basic head for re-identification.
    Args:
        in_channels (int): Number of channels in the input.
        norm_cfg (dict, optional): Configuration of normlization method
            after fc. Defaults to None.
        act_cfg (dict, optional): Configuration of activation method after fc.
            Defaults to None.
        num_classes (int, optional): Number of the identities. Default to None.
        loss (dict, optional): Cross entropy loss to train the
            re-identificaiton module.
        loss_pairwise (dict, optional): Triplet loss to train the
            re-identificaiton module.
        topk (int, optional): Calculate topk accuracy. Default to False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to dict(type='Normal',layer='Linear', mean=0, std=0.01,
            bias=0).
    """

    def __init__(
        self,
        in_channels,
        norm_cfg=None,
        act_cfg=None,
        num_classes=None,
        loss=None,
        loss_pairwise=None,
        topk=(1,),
        init_cfg=dict(type="Normal", layer="Linear", mean=0, std=0.01, bias=0),
    ):
        super(BasicReIDHead, self).__init__(init_cfg)
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk,)
        for _topk in topk:
            assert _topk > 0, "Top-k should be larger than 0"
        self.topk = topk

        if not loss:
            if isinstance(num_classes, int):
                warnings.warn(
                    "Since cross entropy is not set, "
                    "the num_classes will be ignored."
                )
            if not loss_pairwise:
                raise ValueError(
                    "Please choose at least one loss in "
                    "triplet loss and cross entropy loss."
                )
        elif not isinstance(num_classes, int):
            raise TypeError(
                "The num_classes must be a current number, "
                "if there is cross entropy loss."
            )
        self.loss_cls = build_loss(loss) if loss else None
        self.loss_triplet = build_loss(loss_pairwise) if loss_pairwise else None

        self.in_channels = in_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.num_classes = num_classes
        self.accuracy = Accuracy(topk=self.topk)
        self.fp16_enabled = False

        self._init_layers()

    def _init_layers(self):
        """Initialize fc layers."""
        if self.loss_cls:
            self.bn = nn.BatchNorm1d(self.in_channels)
            self.classifier = nn.Linear(self.in_channels, self.num_classes)

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

        # feats = self.pre_logits(x)

        if self.loss_cls:
            feats_bn = self.bn(x)
            cls_score = self.classifier(feats_bn)
            return (x, cls_score)
        return (x,)

    @force_fp32(apply_to=("feats", "cls_score"))
    def loss(self, gt_label, feats, cls_score=None):
        """Compute losses."""
        losses = dict()

        if self.loss_triplet:
            losses["triplet_loss"] = self.loss_triplet(feats, gt_label)

        if self.loss_cls:
            assert cls_score is not None
            losses["ce_loss"] = self.loss_cls(cls_score, gt_label)
            # compute accuracy
            acc = self.accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses["accuracy"] = {f"top-{k}": a for k, a in zip(self.topk, acc)}

        return losses