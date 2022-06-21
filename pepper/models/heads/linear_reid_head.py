#!/usr/bin/env python3

import warnings

import torch.nn as nn

from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmcv.cnn import build_activation_layer, build_norm_layer

from .base_head import BaseHead
from ..builder import HEADS, build_loss
from ..losses import Accuracy

# from mmcls.models.losses import Accuracy


class FcModule(BaseModule):
    """Fully-connected layer module.
    Args:
        in_channels (int): Input channels.
        out_channels (int): Ourput channels.
        norm_cfg (dict, optional): Configuration of normlization method
            after fc. Defaults to None.
        act_cfg (dict, optional): Configuration of activation method after fc.
            Defaults to dict(type='ReLU').
        inplace (bool, optional): Whether inplace the activatation module.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to dict(type='Kaiming', layer='Linear').
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        inplace=True,
        init_cfg=dict(type="Kaiming", layer="Linear"),
    ):
        super(FcModule, self).__init__(init_cfg)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None

        self.fc = nn.Linear(in_channels, out_channels)
        # build normalization layers
        if self.with_norm:
            self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
            self.add_module(self.norm_name, norm)

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_["type"] not in [
                "Tanh",
                "PReLU",
                "Sigmoid",
                "HSigmoid",
                "Swish",
            ]:
                act_cfg_.setdefault("inplace", inplace)
            self.activate = build_activation_layer(act_cfg_)

    @property
    def norm(self):
        """Normalization."""
        return getattr(self, self.norm_name)

    def forward(self, x, activate=True, norm=True):
        """Model forward."""
        x = self.fc(x)
        if norm and self.with_norm:
            x = self.norm(x)
        if activate and self.with_activation:
            x = self.activate(x)
        return x


@HEADS.register_module()
class LinearReIDHead(BaseHead):
    """Linear head for re-identification.
    Args:
        num_fcs (int): Number of fcs.
        in_channels (int): Number of channels in the input.
        fc_channels (int): Number of channels in the fcs.
        out_channels (int): Number of channels in the output.
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
        num_fcs,
        in_channels,
        fc_channels,
        out_channels,
        norm_cfg=None,
        act_cfg=None,
        num_classes=None,
        loss=None,
        loss_pairwise=None,
        topk=(1,),
        init_cfg=dict(type="Normal", layer="Linear", mean=0, std=0.01, bias=0),
    ):
        super(LinearReIDHead, self).__init__(init_cfg)
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

        self.num_fcs = num_fcs
        self.in_channels = in_channels
        self.fc_channels = fc_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.num_classes = num_classes
        self.accuracy = Accuracy(topk=self.topk)
        self.fp16_enabled = False

        self._init_layers()

    def _init_layers(self):
        """Initialize fc layers."""
        self.fcs = nn.ModuleList()

        for i in range(self.num_fcs):
            in_channels = self.in_channels if i == 0 else self.fc_channels
            self.fcs.append(
                FcModule(
                    in_channels, self.fc_channels, self.norm_cfg, self.act_cfg
                )
            )
        in_channels = (
            self.in_channels if self.num_fcs == 0 else self.fc_channels
        )

        self.fc_out = nn.Linear(in_channels, self.out_channels)

        if self.loss_cls:
            self.bn = nn.BatchNorm1d(self.out_channels)
            self.classifier = nn.Linear(self.out_channels, self.num_classes)

    @auto_fp16()
    def forward_train(self, x):
        """Model forward."""
        for m in self.fcs:
            x = m(x)
        feats = self.fc_out(x)
        if self.loss_cls:
            feats_bn = self.bn(feats)
            cls_score = self.classifier(feats_bn)
            return (feats, cls_score)
        return (feats,)

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
