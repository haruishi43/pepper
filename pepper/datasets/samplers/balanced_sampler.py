#!/usr/bin/env python3

"""Balanced Sampler

- Same as NaiveSampler, but this sampler will try to sample pids that have unique
  camids
"""

import copy
import itertools
import warnings
from collections import defaultdict

import numpy as np

from torch.utils.data import DistributedSampler, Sampler

from .utils import reorder_index
from ..builder import SAMPLERS


@SAMPLERS.register_module()
class BalancedIdentitySampler(Sampler):
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
        assert not (
            batch_size % num_instances
        ), "batch_size needs be divisible by num_instances"
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances
        self.batch_size = batch_size
        self.round_up = round_up
        self.shuffle = shuffle

        self._rng = np.random.default_rng(seed)

        # avoid having to run pipelines
        data_infos = copy.deepcopy(
            [info["img_info"] for info in dataset.data_infos]
        )

        # NOTE: important that the index doesn't change!
        self.index_of_pid = defaultdict(list)
        self.camid_of_pid = defaultdict(list)
        for index, info in enumerate(data_infos):
            pid = info["pid"]
            camid = info["camid"]
            self.index_of_pid[pid].append(index)
            self.camid_of_pid[pid].append(camid)

        self.pids = sorted(list(self.index_of_pid.keys()))
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

    def __len__(self):
        # NOTE: changed this from len(self.dataset) since the total number of traning data shrinks
        return self.total_size

    def __iter__(self):
        def remove_same_index(l: list, value):
            assert isinstance(l, list)
            return [i for i, j in enumerate(l) if j != value]

        pid_idxs = list(range(self.num_identities))

        # instances are index of the data
        pid_instances = copy.deepcopy(self.index_of_pid)
        pid_camids = copy.deepcopy(self.camid_of_pid)

        if self.shuffle:
            self._rng.shuffle(pid_idxs)
        else:
            warnings.warn("WARN: `shuffle=False` detected.")

        tot = self.num_iterations * self.num_pids_per_batch
        if self.round_up and self.num_identities % self.num_pids_per_batch != 0:
            # pad
            pid_idxs = (pid_idxs * int(tot / len(pid_idxs) + 1))[:tot]
        elif self.num_identities % self.num_pids_per_batch != 0:
            # drop
            pid_idxs = pid_idxs[: -(len(pid_idxs) % self.num_pids_per_batch)]

        indices = []
        for pid_idx in pid_idxs:
            batch_indices = []

            # obtain pid from index
            pid = self.pids[pid_idx]

            # get some instance
            if len(pid_instances[pid]) == 0:
                # re-add instances if low
                pid_instances[pid] = copy.deepcopy(self.index_of_pid[pid])
                pid_camids[pid] = copy.deepcopy(self.camid_of_pid[pid])

            instances = pid_instances[pid]  # List
            camids = pid_camids[pid]  # List
            assert len(instances) == len(camids)

            if self.shuffle:
                instance_idx = self._rng.integers(len(instances))
                instance = instances.pop(instance_idx)
                camid = camids.pop(instance_idx)
            else:
                default_idx = 0
                instance = instances.pop(default_idx)
                camid = camids.pop(default_idx)

            # add to batch
            batch_indices.append(instance)

            # select camids that are different
            # note that the indices should match `instances`
            select_cams_indices = remove_same_index(camids, camid)

            if select_cams_indices:
                # sample from different camids
                if len(select_cams_indices) >= self.num_instances:
                    use_indices = self._rng.choice(
                        select_cams_indices,
                        size=self.num_instances - 1,
                        replace=False,
                    )
                else:
                    use_indices = self._rng.choice(
                        select_cams_indices,
                        size=self.num_instances - 1,
                        replace=True,
                    )
                for i in use_indices:
                    batch_indices.append(instances[i])
                use_indices = set(use_indices)
            else:
                # sample from remaining instances (that have the same camid)
                if len(instances) == 0:
                    use_indices = []
                    # only one image for this identity
                    batch_indices += [instance] * (self.num_instances - 1)
                elif len(instances) >= self.num_instances:
                    use_indices = self._rng.choice(
                        list(range(len(instances))),
                        size=self.num_instances - 1,
                        replace=False,
                    )
                    # sample from remaining instances
                    for i in use_indices:
                        batch_indices.append(instances[i])
                else:
                    # when the number of instances are low, we repeat some
                    use_indices = []
                    batch_indices += list(
                        self._rng.choice(
                            instances,
                            size=self.num_instances - 1,
                            replace=True,
                        )
                    )

            # remove used indices
            if len(use_indices) == 0:
                pid_instances[pid] = []
                pid_camids[pid] = []
            else:
                for i in sorted(use_indices, reverse=True):
                    # we remove from both instances and camids
                    instances.pop(i)
                    camids.pop(i)

            indices += batch_indices

        assert (
            len(indices) == self.total_size
        ), f"indices={len(indices)}, should be {self.total_size}"

        return iter(indices)


@SAMPLERS.register_module()
class BalancedIdentityDistributedSampler(DistributedSampler):
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

        # NOTE: important that the index doesn't change!
        self.index_of_pid = defaultdict(list)
        self.camid_of_pid = defaultdict(list)
        for index, info in enumerate(data_infos):
            pid = info["pid"]
            camid = info["camid"]
            self.index_of_pid[pid].append(index)
            self.camid_of_pid[pid].append(camid)

        self.pids = sorted(list(self.index_of_pid.keys()))
        self.num_identities = len(self.pids)

        # num_samples: the number of samples for each replicas

        # FIXME: do we need to overwrite this?

        # FIXME: num_iterations should not depend on identities!
        if self.round_up and self.num_identities % self.num_pids_per_batch != 0:
            self.num_iterations = (
                self.num_identities // self.num_pids_per_batch + 1
            )
        else:
            self.num_iterations = self.num_identities // self.num_pids_per_batch

        self.total_size = (
            self.num_iterations * self.num_pids_per_batch * self.num_instances
        )

        # HACK: change the number of iterations (len(dataloader) calls len(sampler))
        self.num_samples = self.total_size // self.num_replicas

    def __iter__(self):

        self._rng = np.random.default_rng(self.seed + self.epoch)

        def remove_same_index(l: list, value):
            assert isinstance(l, list)
            return [i for i, j in enumerate(l) if j != value]

        pid_idxs = list(range(self.num_identities))

        # instances are index of the data
        pid_instances = copy.deepcopy(self.index_of_pid)
        pid_camids = copy.deepcopy(self.camid_of_pid)

        if self.shuffle:
            self._rng.shuffle(pid_idxs)
        else:
            warnings.warn("WARN: `shuffle=False` detected.")

        tot = self.num_iterations * self.num_pids_per_batch
        if self.round_up and self.num_identities % self.num_pids_per_batch != 0:
            # pad
            pid_idxs = (pid_idxs * int(tot / len(pid_idxs) + 1))[:tot]
        elif self.num_identities % self.num_pids_per_batch != 0:
            # drop
            pid_idxs = pid_idxs[: -(len(pid_idxs) % self.num_pids_per_batch)]

        indices = []
        for pid_idx in pid_idxs:
            batch_indices = []

            # obtain pid from index
            pid = self.pids[pid_idx]

            # get some instance
            if len(pid_instances[pid]) == 0:
                # re-add instances if low
                pid_instances[pid] = copy.deepcopy(self.index_of_pid[pid])
                pid_camids[pid] = copy.deepcopy(self.camid_of_pid[pid])

            instances = pid_instances[pid]  # List
            camids = pid_camids[pid]  # List
            assert len(instances) == len(camids)

            if self.shuffle:
                instance_idx = self._rng.integers(len(instances))
                instance = instances.pop(instance_idx)
                camid = camids.pop(instance_idx)
            else:
                default_idx = 0
                instance = instances.pop(default_idx)
                camid = camids.pop(default_idx)

            # add to batch
            batch_indices.append(instance)

            # select camids that are different
            # note that the indices should match `instances`
            select_cams_indices = remove_same_index(camids, camid)

            if select_cams_indices:
                # sample from different camids
                if len(select_cams_indices) >= self.num_instances:
                    use_indices = self._rng.choice(
                        select_cams_indices,
                        size=self.num_instances - 1,
                        replace=False,
                    )
                else:
                    use_indices = self._rng.choice(
                        select_cams_indices,
                        size=self.num_instances - 1,
                        replace=True,
                    )
                for i in use_indices:
                    batch_indices.append(instances[i])
                use_indices = set(use_indices)
            else:
                # sample from remaining instances (that have the same camid)
                if len(instances) == 0:
                    use_indices = []
                    # only one image for this identity
                    batch_indices += [instance] * (self.num_instances - 1)
                elif len(instances) >= self.num_instances:
                    use_indices = self._rng.choice(
                        list(range(len(instances))),
                        size=self.num_instances - 1,
                        replace=False,
                    )
                    # sample from remaining instances
                    for i in use_indices:
                        batch_indices.append(instances[i])
                else:
                    # when the number of instances are low, we repeat some
                    use_indices = []
                    batch_indices += list(
                        self._rng.choice(
                            instances,
                            size=self.num_instances - 1,
                            replace=True,
                        )
                    )

            # remove used indices
            if len(use_indices) == 0:
                pid_instances[pid] = []
                pid_camids[pid] = []
            else:
                for i in sorted(use_indices, reverse=True):
                    # we remove from both instances and camids
                    instances.pop(i)
                    camids.pop(i)

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
