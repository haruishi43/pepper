#!/usr/bin/env python3

import torch
import torch.nn as nn

from mmcv.runner import auto_fp16, force_fp32

from .base_head import BaseHead
from .utils import weights_init_classifier, weights_init_kaiming
from ..builder import HEADS, build_loss
from ..losses import Accuracy


@HEADS.register_module()
class BasicHead(BaseHead):
    """Basic head for re-identification.
    Args:
        in_channels (int): Number of channels in the input.
        norm_cfg (dict, optional): Configuration of normlization method
            after fc. Defaults to None.
        act_cfg (dict, optional): Configuration of activation method after fc.
            Defaults to None.
        num_classes (int, optional): Number of the identities. Default to None.
        loss_cls (list, tuple, dict, optional): Cross entropy loss to train the
            re-identificaiton module.
        loss_pairwise (list, tuple, dict, optional): Triplet loss to train the
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
        loss_cls=None,
        loss_pairwise=None,
        topk=(1,),
        init_cfg=dict(type="Normal", layer="Linear", mean=0, std=0.01, bias=0),
    ):
        super().__init__(init_cfg)
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk,)
        for _topk in topk:
            assert _topk > 0, "Top-k should be larger than 0"
        self.topk = topk

        # Setup losses
        if not loss_cls:
            self.loss_cls = None
            if not loss_pairwise:
                raise ValueError(
                    "Please choose at least one loss in "
                    "loss_cls and loss_pairwise."
                )
        else:
            if not isinstance(num_classes, int):
                raise TypeError(
                    "The num_classes must be a current number, "
                    "if there is cross entropy loss."
                )

            if isinstance(loss_cls, dict):
                self.loss_cls = build_loss(loss_cls)
            elif isinstance(loss_cls, (list, tuple)):
                self.loss_cls = nn.ModuleList()
                for loss in loss_cls:
                    self.loss_cls.append(build_loss(loss))
            else:
                raise TypeError(
                    f"loss_cls must be a dict or sequence of dict,\
                    but got {type(loss_cls)}"
                )

        if not loss_pairwise:
            self.loss_pairwise = None
        else:
            if isinstance(loss_pairwise, dict):
                self.loss_pairwise = build_loss(loss_pairwise)
            elif isinstance(loss_pairwise, (list, tuple)):
                self.loss_pairwise = nn.ModuleList()
                for loss in loss_pairwise:
                    self.loss_pairwise.append(build_loss(loss))
            else:
                raise TypeError(
                    f"loss_pairwise must be a dict or sequence of dict,\
                    but got {type(loss_pairwise)}"
                )

        self.in_channels = in_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.num_classes = num_classes
        self.accuracy = Accuracy(topk=self.topk)
        self.fp16_enabled = False

        self._init_layers()

    def _init_layers(self):
        """Initialize layers"""
        if self.loss_cls:
            self.bn = nn.BatchNorm1d(self.in_channels)
            self.bn.bias.requires_grad_(False)
            self.classifier = nn.Linear(
                self.in_channels, self.num_classes, bias=False
            )

            self.bn.apply(weights_init_kaiming)
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

        return x

    @auto_fp16()
    def forward_train(self, x):
        """Model forward."""

        if self.loss_cls:
            feats_bn = self.bn(x)
            cls_score = self.classifier(feats_bn)
            return (x, cls_score)
        return (x,)

    @force_fp32(apply_to=("feats", "cls_score"))
    def loss(self, gt_label, feats, cls_score=None):
        """Compute losses."""
        losses = dict()

        # Classification (softmax) losses
        if self.loss_cls:
            assert cls_score is not None
            if not isinstance(self.loss_cls, nn.ModuleList):
                loss_cls = [self.loss_cls]
            else:
                loss_cls = self.loss_cls
            for lc in loss_cls:
                if lc.loss_name not in losses:
                    losses[lc.loss_name] = lc(
                        cls_score=cls_score,
                        label=gt_label,
                    )
                else:
                    losses[lc.loss_name] += lc(
                        cls_score=cls_score,
                        label=gt_label,
                    )

            # compute accuracy
            acc = self.accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses["accuracy"] = {f"top-{k}": a for k, a in zip(self.topk, acc)}

        # Metric (distance) losses
        if self.loss_pairwise:
            if not isinstance(self.loss_pairwise, nn.ModuleList):
                loss_pairwise = [self.loss_pairwise]
            else:
                loss_pairwise = self.loss_pairwise
            for lp in loss_pairwise:
                if lp.loss_name not in losses:
                    losses[lp.loss_name] = lp(
                        inputs=feats,
                        targets=gt_label,
                    )
                else:
                    losses[lp.loss_name] += lp(
                        inputs=feats,
                        targets=gt_label,
                    )

        return losses
