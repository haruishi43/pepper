#!/usr/bin/env python3

import torch
import torch.nn as nn

from mmcv.cnn import build_activation_layer, build_norm_layer

from ..builder import NECKS


@NECKS.register_module()
class PartPooling(nn.Module):
    def __init__(self, num_parts=6):
        super().__init__()
        assert num_parts > 0
        self.num_parts = num_parts
        self.pp = nn.AdaptiveAvgPool2d((num_parts, 1))

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            # multiple side outputs
            outs = tuple([self.pp(x) for x in inputs])
            outs = tuple(
                [
                    out.view(x.size(0), x.size(1), -1)
                    for out, x in zip(outs, inputs)
                ]
            )
        elif isinstance(inputs, torch.Tensor):
            outs = self.pp(inputs)
            outs = outs.view(inputs.size(0), inputs.size(1), -1)
        else:
            raise TypeError("neck input shuld be tuple or torch.tensor")
        return outs


class RPPModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_parts=6,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        act_cfg=dict(type="ReLU"),
    ):
        super().__init__()
        self.num_parts = num_parts

        self.add_block = nn.Conv2d(
            in_channels, num_parts, kernel_size=1, bias=False
        )

        _, self.norm = build_norm_layer(norm_cfg, in_channels)
        self.act = build_activation_layer(act_cfg)

        self.softmax = nn.Softmax(dim=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        w = self.add_block(x)
        p = self.softmax(w)
        y = []
        for i in range(self.num_parts):
            p_i = p[:, i, :, :]
            p_i = torch.unsqueeze(p_i, 1)
            y_i = torch.mul(x, p_i)
            y_i = self.act(self.norm(y_i))
            y_i = self.gap(y_i)
            y.append(y_i)

        f = torch.cat(y, 2)
        f = f.view(x.size(0), x.size(1), -1)
        return f


@NECKS.register_module()
class RefinedPartPooling(nn.Module):
    def __init__(
        self,
        in_channels,
        num_parts=6,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        act_cfg=dict(type="ReLU"),
    ) -> None:
        super().__init__()

        if isinstance(in_channels, int):
            in_channels = [in_channels]
        assert isinstance(
            in_channels, (tuple, list)
        ), f"in_channels should not be {type(in_channels)}"
        assert len(in_channels) > 0

        blocks = []
        for in_channel in in_channels:
            blocks.append(
                RPPModule(
                    in_channels=in_channel,
                    num_parts=num_parts,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def init_weights(self):
        for b in self.blocks:
            b.init_weights()

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            assert len(inputs) == len(
                self.blocks
            ), f"expected {len(self.blocks)} but got {len(inputs)}"
        elif isinstance(inputs, torch.tensor):
            assert len(self.blocks) == 1
            inputs = (inputs,)
        else:
            raise TypeError("neck inputs should be tuple or torch.tensor")

        outs = []
        for input, block in zip(inputs, self.blocks):
            outs.append(block(input))
        outs = tuple(outs)
        return outs
