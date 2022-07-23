#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import auto_fp16, force_fp32

from mmcls.models.losses import Accuracy
from mmcls.models.builder import HEADS
from mmcls.models.utils import is_tracing
from mmcls.models.heads.base_head import BaseHead

# NOTE: attributes for losses are different! (don't build mmcls losses)
from pepper.models.builder import METRIC_LINEAR_LAYERS, build_loss


@HEADS.register_module()
class MetricHead(BaseHead):
    """classification + pairwise head.
    Args:
        loss (dict): Config of classification loss.
        topk (int | tuple): Top-k accuracy.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use Mixup/CutMix or something like that during training,
            it is not reasonable to calculate accuracy. Defaults to False.
    """

    def __init__(
        self,
        in_channels,
        norm_cfg=None,
        act_cfg=None,
        num_classes=None,
        loss_cls=None,
        loss_pairwise=None,
        linear_layer=None,
        topk=(1,),
        init_cfg=dict(type="Normal", layer="Linear", mean=0, std=0.01, bias=0),
    ):
        super(MetricHead, self).__init__(init_cfg=init_cfg)
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

        # should be set to None when we don't use metric learning
        self.linear_cfg = linear_layer

        self._init_layers()

    def _init_layers(self):
        """Initialize layers"""

        self.bn = nn.BatchNorm1d(self.in_channels)
        self.bn.bias.requires_grad_(False)

        self.classifier = self.create_classification_layer(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            linear_cfg=self.linear_cfg,
        )

    @staticmethod
    def create_classification_layer(in_channels, num_classes, linear_cfg=None):
        """Function for creating the last linear layer for classsification"""
        if linear_cfg is None:
            # if the recipe for the linear layer is not specified, we return the default nn.Linear
            return nn.Linear(in_channels, num_classes, bias=False)
        else:
            defaults = dict(
                in_channels=in_channels,
                num_classes=num_classes,
            )
            linear_cfg.update(defaults)
            return METRIC_LINEAR_LAYERS.build(linear_cfg)

    @auto_fp16()
    def pre_logits(self, x, gt_label):

        # deals with tuple outputs from previous layers
        if isinstance(x, tuple):
            if len(x) > 1:
                # use the last one
                x = x[-1]
            else:
                x = x[0]

        assert isinstance(x, torch.Tensor)

        feats_bn = self.bn(x)

        if self.linear_cfg is None:
            cls_score = self.classifier(feats_bn)
        else:
            cls_score = self.classifier(feats_bn, gt_label)

        return dict(
            cls_score=cls_score,
            feats=x,
        )

    def forward_train(self, x, gt_label, **kwargs):
        """Model forward."""

        if isinstance(x, dict):
            # for metric learning we use dict
            cls_score = x.get("cls_score", None)
            feats = x.get("feats", None)
        else:
            # assert that cls_score is the main output
            # backward compatibility
            cls_score = x

        if isinstance(cls_score, (tuple, list)):
            cls_score = cls_score[-1]
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]

        return self.loss(gt_label=gt_label, cls_score=cls_score, feats=feats)

    @force_fp32(apply_to=("feats", "cls_score"))
    def loss(self, gt_label, cls_score, feats=None):
        """Compute losses."""
        losses = dict()

        if cls_score is not None:
            # compute accuracy
            acc = self.accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses["accuracy"] = {f"top-{k}": a for k, a in zip(self.topk, acc)}

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

        # Metric (distance) losses
        if self.loss_pairwise:
            assert feats is not None
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

    def simple_test(
        self,
        x,
        softmax=True,
        post_process=True,
        return_feats=False,
        **kwargs,
    ):
        if isinstance(x, dict):
            cls_score = x.get("cls_score", None)
            feats = x.get("feats", None)

            assert cls_score is not None, "could not return cls_score"
        else:
            cls_score = x
            feats = None

        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None
            )
        else:
            pred = cls_score

        if post_process:
            pred = self.post_process(pred)
            if feats is not None:
                feats = self.post_process(feats)

        if return_feats:
            return pred, feats
        else:
            return pred

    def post_process(self, pred):
        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred
