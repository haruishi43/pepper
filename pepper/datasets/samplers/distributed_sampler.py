#!/usr/bin/env python3

import torch
from torch.utils.data import DistributedSampler as _DistributedSampler

from ..builder import SAMPLERS


@SAMPLERS.register_module()
class DistributedSampler(_DistributedSampler):
    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        seed=0,
        shuffle=True,
        round_up=True,
        is_val=False,
    ):
        # subclass: https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler

        self.is_val = is_val
        if is_val:
            assert not shuffle, "no shuffle for validation"
            assert round_up, "we need all validation data"

        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )
        self.round_up = round_up
        if self.round_up:
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.dataset)

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (indices * int(self.total_size / len(indices) + 1))[
                : self.total_size
            ]
        assert len(indices) == self.total_size

        # subsample
        # if self.is_val:
        #     indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        # else:
        indices = indices[self.rank : self.total_size : self.num_replicas]

        # last checks
        if self.round_up:
            assert len(indices) == self.num_samples

        return iter(indices)
