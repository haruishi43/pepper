#!/usr/bin/env python3

from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn


class BaseTemporalLayer(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def _forward(self, x, **kwargs):
        pass

    def forward(self, x, **kwargs):

        # TODO:
        # - what to do for multi-layer inputs (tuple of tensors)

        # deals with tuple outputs from previous layers
        if isinstance(x, tuple):
            if len(x) > 1:
                # use the last one
                x = x[-1]
            else:
                x = x[0]

        assert isinstance(x, torch.Tensor)

        return self._forward(x, **kwargs)
