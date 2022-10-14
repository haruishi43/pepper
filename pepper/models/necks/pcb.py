#!/usr/bin/env python3

import torch
import torch.nn as nn

from ..builder import NECKS


@NECKS.register_module()
class PartPooling(nn.Module):
    def __init__(self, num_parts=6, refined=False):
        super().__init__()

        self.nparts = num_parts
