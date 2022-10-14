#!/usr/bin/env python3

from .base_head import BaseHead
from ..builder import HEADS


@HEADS.register_module()
class PCBHead(BaseHead):
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
