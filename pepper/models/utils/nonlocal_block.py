#!/usr/bin/env python3

import torch.nn as nn

from mmcv.cnn.bricks import NonLocal2d
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS


@PLUGIN_LAYERS.register_module()
class NonLocalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        num_layers,
        reduction=2,
        sub_sample=False,
        mode="dot_product",
        use_scale=True,
        conv_cfg=dict(type="Conv2d"),
        norm_cfg=dict(type="BN", requires_grad=True),
        **kwargs,
    ):
        """Reproduce Non-Local blocks for FastReid

        - needed `num_layers` since there are more than 1 non local layers per stage
        """
        super().__init__()

        self.layers = []

        for i in range(num_layers):
            layer = NonLocal2d(
                in_channels=in_channels,
                sub_sample=sub_sample,
                reduction=reduction,
                use_scale=use_scale,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                mode=mode,
                **kwargs,
            )
            layer_name = f"layer{i+1}"
            self.add_module(layer_name, layer)
            self.layers.append(layer_name)

    def forward(self, x):
        for layer_name in self.layers:
            layer = getattr(self, layer_name)
            x = layer(x)
        return x
