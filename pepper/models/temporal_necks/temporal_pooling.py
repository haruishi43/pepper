#!/usr/bin/env python3

import torch

from .base import BaseTemporalLayer
from ..builder import TEMPORAL


@TEMPORAL.register_module()
class TemporalPooling(BaseTemporalLayer):

    _pool_types = ("mean", "median")

    def __init__(self, pooling_method="mean"):
        super().__init__()

        assert (
            pooling_method in self._pool_types
        ), f"pooling_method must be in {self._pool_types}"
        self.method = pooling_method

    def _forward(self, x, **kwargs):
        if self.method == "mean":
            x = torch.mean(x, dim=1)
        elif self.method == "median":
            x = torch.median(x, dim=1)
        return x
