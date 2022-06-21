#!/usr/bin/env python3

"""Naive Identity Sampler

- Samples identities given batch size and number of instances
- Other properties such as camids are not taken into consideration
"""

from collections import defaultdict
import copy
import itertools
import warnings

import numpy as np

import torch
from torch.utils.data import DistributedSampler, Sampler

from .utils import reorder_index
from ..builder import SAMPLERS


@SAMPLERS.register_module()
class NaiveIdentitySampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        num_instances: int = 4,
        seed: int = 0,
        shuffle: bool = True,
        round_up: bool = True,
    ) -> None:
        self.dataset = dataset
        assert not (batch_size > len(dataset))
        assert (
            batch_size % num_instances == 0
        ), "batch_size needs be divisible by num_instances"
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances
        self.batch_size = batch_size
        self.round_up = round_up
        self.shuffle = shuffle  # NOTE: shuffle should be True

        # use it's own random number generator for reproducibility
        self._rng = np.random.default_rng(seed)

        # avoid having to run pipelines
        data_infos = copy.deepcopy(
            [info["img_info"] for info in dataset.data_infos]
        )

        self.pid_index = defaultdict(list)
        for index, info in enumerate(data_infos):
            pid = info["pid"]
            self.pid_index[pid].append(index)

        self.pids = sorted(list(self.pid_index.keys()))
        self.num_identities = len(self.pids)

        # `num_iterations` is the number of iterations during the epoch
        # note that each iteration produces a batch
        # each epoch should show every identity atleat once
        if self.round_up and self.num_identities % self.num_pids_per_batch != 0:
            self.num_iterations = (
                self.num_identities // self.num_pids_per_batch + 1
            )
        else:
            self.num_iterations = self.num_identities // self.num_pids_per_batch

        self.total_size = (
            self.num_iterations * self.num_pids_per_batch * self.num_instances
        )

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        available_pids = copy.deepcopy(self.pids)
        pid_idxs = copy.deepcopy(self.pid_index)
        removed_pids = []

        if self.shuffle:
            self._rng.shuffle(available_pids)
        else:
            warnings.warn("WARN: `shuffle=False` detected.")

        indices = []
        for _ in range(self.num_iterations):
            batch_indices = []

            if len(available_pids) < self.num_pids_per_batch:
                # we need to add extra pids from `removed`
                num_add = self.num_pids_per_batch - len(available_pids)
                # add first couple removed pids
                available_pids.extend(removed_pids[:num_add])

            if self.shuffle:
                selected_pids = self._rng.choice(
                    available_pids,
                    self.num_pids_per_batch,
                    replace=False,
                ).tolist()
            else:
                selected_pids = available_pids[: self.num_pids_per_batch]

            for pid in selected_pids:

                # if pid was removed, add the indices back
                if pid not in pid_idxs.keys():
                    pid_idxs[pid] = copy.deepcopy(self.pid_index[pid])

                idxs = pid_idxs[pid]
                if self.shuffle:
                    if len(idxs) < self.num_instances:
                        idxs = self._rng.choice(
                            idxs,
                            size=self.num_instances,
                            replace=True,
                        ).tolist()
                    self._rng.shuffle(idxs)
                else:
                    if len(idxs) < self.num_instances:
                        idxs = (idxs * int(self.num_instances / len(idxs) + 1))[
                            : self.num_instances
                        ]

                for _ in range(self.num_instances):
                    batch_indices.append(idxs.pop(0))

                # remove pids if the number of indices remaining are low
                if len(idxs) < self.num_instances:
                    pid_idxs.pop(pid)

                # remove after use
                available_pids.remove(pid)
                removed_pids.append(pid)

            assert len(batch_indices) == self.batch_size
            indices += batch_indices

        assert (
            len(indices) == self.total_size
        ), f"indices={len(indices)}, should be {self.total_size}"

        return iter(indices)


@SAMPLERS.register_module()
class NaiveIdentityDistributedSampler(DistributedSampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - dataset (Dataset)
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        batch_size=32,
        num_instances=4,
        seed=0,
        shuffle=True,
        round_up=True,
    ) -> None:
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )
        assert not (batch_size > len(dataset))
        assert not (
            batch_size % num_instances
        ), "batch_size needs be divisible by num_instances"
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances
        self.batch_size = batch_size
        self.round_up = round_up

        self._rng = np.random.default_rng(self.seed)

        # avoid having to run pipelines
        data_infos = copy.deepcopy(
            [info["img_info"] for info in dataset.data_infos]
        )

        self.pid_index = defaultdict(list)
        for index, info in enumerate(data_infos):
            pid = info["pid"]
            # camid = info["camid"]
            self.pid_index[pid].append(index)

        self.pids = sorted(list(self.pid_index.keys()))
        self.num_identities = len(self.pids)

        if self.round_up and self.num_identities % self.num_pids_per_batch != 0:
            self.num_iterations = (
                self.num_identities // self.num_pids_per_batch + 1
            )
        else:
            self.num_iterations = self.num_identities // self.num_pids_per_batch

        self.total_size = (
            self.num_iterations * self.num_pids_per_batch * self.num_instances
        )

    def __iter__(self):
        available_pids = copy.deepcopy(self.pids)
        pid_idxs = copy.deepcopy(self.pid_index)
        removed_pids = []

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            self._rng = np.random.default_rng(self.seed + self.epoch)
            pid_indices = torch.randperm(
                len(available_pids), generator=g
            ).tolist()
            available_pids = [available_pids[i] for i in pid_indices]

        indices = []
        for _ in range(self.num_iterations):
            batch_indices = []

            if len(available_pids) < self.num_pids_per_batch:
                # we need to add extra pids from `removed`
                num_add = self.num_pids_per_batch - len(available_pids)
                available_pids.extend(removed_pids[:num_add])

            if self.shuffle:
                selected_pids = self._rng.choice(
                    available_pids,
                    self.num_pids_per_batch,
                    replace=False,
                ).tolist()
            else:
                selected_pids = available_pids[: self.num_pids_per_batch]

            for pid in selected_pids:
                # if pid was removed, add the indices back
                if pid not in pid_idxs.keys():
                    pid_idxs[pid] = copy.deepcopy(self.pid_index[pid])

                idxs = pid_idxs[pid]
                if self.shuffle:
                    if len(idxs) < self.num_instances:
                        idxs = self._rng.choice(
                            idxs,
                            size=self.num_instances,
                            replace=True,
                        ).tolist()
                    self._rng.shuffle(idxs)
                else:
                    if len(idxs) < self.num_instances:
                        idxs = (idxs * int(self.num_instances / len(idxs) + 1))[
                            : self.num_instances
                        ]

                for _ in range(self.num_instances):
                    batch_indices.append(idxs.pop(0))

                # remove pids if the number of indices remaining are low
                if len(idxs) < self.num_instances:
                    pid_idxs.pop(pid)

                # remove after use
                available_pids.remove(pid)
                removed_pids.append(pid)

            assert len(batch_indices) == self.batch_size
            indices += batch_indices

        assert (
            len(indices) == self.total_size
        ), f"indices={len(indices)}, should be {self.total_size}"

        # print("before:", len(indices), indices)
        indices = reorder_index(indices, self.num_replicas)
        # TODO: add checks?
        # print("after", len(indices), indices)
        indices = itertools.islice(indices, self.rank, None, self.num_replicas)
        return iter(indices)
