#!/usr/bin/env python3

import torch
import torch.nn as nn


class TemporalPooling(nn.Module):

    _pool_types = "avg"

    def __init__(self, pooling_method="avg"):
        super().__init__()

        assert (
            pooling_method in self._pool_types
        ), f"pooling_method must be in {self._pool_types}"

        if pooling_method == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError()

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            inputs = tuple([x.transpose(-1, -2) for x in inputs])
            # inputs = tuple([x.permute(0, 2, 1) for x in inputs])
            outs = tuple([self.pool(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)]
            )
        elif isinstance(inputs, torch.Tensor):
            inputs = inputs.transpose(-1, -2)
            # inputs = inputs.permute(0, 2, 1)
            outs = self.pool(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError("neck inputs should be tuple or torch.tensor")
        return outs
