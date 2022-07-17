#!/usr/bin/env python3

import torch
import torch.nn as nn

from mmcv.runner import BaseModule, auto_fp16
from mmcv.cnn import build_activation_layer, build_norm_layer

from .basic_head import BasicHead
from ..builder import HEADS


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
class LinearHead(BasicHead):
    """Linear head for re-identification.
    Args:
        num_fcs (int): Number of fcs.
        fc_channels (int): Number of channels in the fcs.
        out_channels (int): Number of channels in the output.
    """

    def __init__(
        self,
        num_fcs,
        out_channels,
        fc_channels,
        **kwargs,
    ):
        self.num_fcs = num_fcs
        self.fc_channels = fc_channels
        self.out_channels = out_channels

        super(LinearHead, self).__init__(**kwargs)

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
            self.bn.bias.requires_grad_(False)
            self.classifier = nn.Linear(
                self.out_channels, self.num_classes, bias=False
            )

        # TODO add weight initialization for layers

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

        for m in self.fcs:
            x = m(x)
        feats = self.fc_out(x)

        return feats

    @auto_fp16()
    def forward_train(self, x):
        """Model forward."""

        # feats = self.pre_logits(x)

        if self.loss_cls:
            feats_bn = self.bn(x)
            cls_score = self.classifier(feats_bn)
            return (x, cls_score)
        return (x,)
