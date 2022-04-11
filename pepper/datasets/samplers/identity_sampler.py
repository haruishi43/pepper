#!/usr/bin/env python3

from collections import defaultdict
import copy
import itertools

import numpy as np

import torch
from torch.utils.data import DistributedSampler, Sampler

from ..builder import SAMPLERS


def no_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


def reorder_index(batch_indices, world_size):
    """Reorder indices of samples to align with DataParallel training.
    In this order, each process will contain all images for one ID, triplet loss
    can be computed within each process, and BatchNorm will get a stable result.
    Args:
        batch_indices: A batched indices generated by sampler
        world_size: number of process
    Returns:
    """
    mini_batchsize = len(batch_indices) // world_size
    reorder_indices = []
    for i in range(0, mini_batchsize):
        for j in range(0, world_size):
            reorder_indices.append(batch_indices[i + j * mini_batchsize])
    return reorder_indices


@SAMPLERS.register_module()
class NaiveIdentitySampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size=32,
        num_instances=4,
        shuffle=False,
        seed=0,
        round_up=True,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed  # not used

        assert not (batch_size > len(dataset))
        assert not (
            batch_size % num_instances
        ), "batch_size needs be divisible by num_instances"
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances
        self.batch_size = batch_size
        self.round_up = round_up

        # avoid having to run pipelines
        data_infos = copy.deepcopy(
            [info["sampler_info"] for info in dataset.data_infos]
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

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        available_pids = copy.deepcopy(self.pids)
        removed_pids = []

        if self.shuffle:
            pid_indices = torch.randperm(len(available_pids)).tolist()
            available_pids = [available_pids[i] for i in pid_indices]

        batch_idxs_dict = {}
        indices = []
        for _ in range(self.num_iterations):
            batch_indices = []

            if len(available_pids) < self.num_pids_per_batch:
                # we need to add extra pids from `removed`
                num_add = self.num_pids_per_batch - len(available_pids)
                # shuffle?
                available_pids.extend(removed_pids[:num_add])

            if self.shuffle:
                selected_pids = np.random.choice(
                    available_pids,
                    self.num_pids_per_batch,
                    replace=False,
                ).tolist()
            else:
                selected_pids = available_pids[: self.num_pids_per_batch]

            for pid in selected_pids:
                # Register pid in batch_idxs_dict if not
                if pid not in batch_idxs_dict:
                    idxs = copy.deepcopy(self.pid_index[pid])

                    if self.shuffle:
                        if len(idxs) < self.num_instances:
                            idxs = np.random.choice(
                                idxs,
                                size=self.num_instances,
                                replace=True,
                            ).tolist()
                        np.random.shuffle(idxs)
                    else:
                        if len(idxs) < self.num_instances:
                            idxs = (
                                idxs * int(self.num_instances / len(idxs) + 1)
                            )[: self.num_instances]
                    batch_idxs_dict[pid] = idxs

                avl_idxs = batch_idxs_dict[pid]
                for _ in range(self.num_instances):
                    batch_indices.append(avl_idxs.pop(0))

                if len(avl_idxs) < self.num_instances:
                    available_pids.remove(pid)
                    batch_idxs_dict.pop(pid)
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
        seed=0,
        shuffle=True,
        batch_size=32,
        num_instances=4,
        round_up=True,
    ):
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

        # avoid having to run pipelines
        data_infos = copy.deepcopy(
            [info["sampler_info"] for info in dataset.data_infos]
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
        removed_pids = []

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            pid_indices = torch.randperm(
                len(available_pids), generator=g
            ).tolist()
            available_pids = [available_pids[i] for i in pid_indices]

        batch_idxs_dict = {}
        indices = []
        for _ in range(self.num_iterations):
            batch_indices = []

            if len(available_pids) < self.num_pids_per_batch:
                # we need to add extra pids from `removed`
                num_add = self.num_pids_per_batch - len(available_pids)
                # shuffle?
                available_pids.extend(removed_pids[:num_add])

            if self.shuffle:
                selected_pids = np.random.choice(
                    available_pids,
                    self.num_pids_per_batch,
                    replace=False,
                ).tolist()
            else:
                selected_pids = available_pids[: self.num_pids_per_batch]

            for pid in selected_pids:
                # Register pid in batch_idxs_dict if not
                if pid not in batch_idxs_dict:
                    idxs = copy.deepcopy(self.pid_index[pid])

                    if self.shuffle:
                        if len(idxs) < self.num_instances:
                            idxs = np.random.choice(
                                idxs,
                                size=self.num_instances,
                                replace=True,
                            ).tolist()
                        np.random.shuffle(idxs)
                    else:
                        if len(idxs) < self.num_instances:
                            idxs = (
                                idxs * int(self.num_instances / len(idxs) + 1)
                            )[: self.num_instances]
                    batch_idxs_dict[pid] = idxs

                avl_idxs = batch_idxs_dict[pid]
                for _ in range(self.num_instances):
                    batch_indices.append(avl_idxs.pop(0))

                if len(avl_idxs) < self.num_instances:
                    available_pids.remove(pid)
                    batch_idxs_dict.pop(pid)
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


@SAMPLERS.register_module()
class BalancedIdentitySampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size=32,
        num_instances=4,
        shuffle=True,
        seed=0,
        round_up=True,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = 0

        assert not (batch_size > len(dataset))
        assert not (
            batch_size % num_instances
        ), "batch_size needs be divisible by num_instances"
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances
        self.batch_size = batch_size
        self.round_up = round_up

        # avoid having to run pipelines
        self.data_infos = copy.deepcopy(
            [info["sampler_info"] for info in dataset.data_infos]
        )

        self.index_pid = dict()
        self.pid_index = defaultdict(list)
        self.pid_cam = defaultdict(list)
        for index, info in enumerate(self.data_infos):
            pid = info["pid"]
            camid = info["camid"]
            self.index_pid[index] = pid
            self.pid_index[pid].append(index)
            self.pid_cam[pid].append(camid)

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

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        if self.shuffle:
            identities = torch.randperm(self.num_identities).tolist()
        else:
            identities = torch.arange(self.num_identities).tolist()

        tot = self.num_iterations * self.num_pids_per_batch
        if self.round_up and len(identities) % self.num_pids_per_batch != 0:
            # pad
            identities = (identities * int(tot / len(identities) + 1))[:tot]
        elif len(identities) % self.num_pids_per_batch != 0:
            # drop
            identities = identities[
                : -(len(identities) % self.num_pids_per_batch)
            ]

        indices = []
        for i in identities:
            batch_indices = []

            pid_index = np.random.choice(self.pid_index[self.pids[i]])
            batch_indices.append(pid_index)
            data = self.data_infos[pid_index]
            pid = data["pid"]
            camid = data["camid"]

            same_pid_indices = self.pid_index[pid]
            cam_list = self.pid_cam[pid]

            select_cams = no_index(cam_list, camid)

            if select_cams:
                if len(select_cams) >= self.num_instances:
                    cam_indices = np.random.choice(
                        select_cams,
                        size=self.num_instances - 1,
                        replace=False,
                    )
                else:
                    cam_indices = np.random.choice(
                        select_cams,
                        size=self.num_instances - 1,
                        replace=True,
                    )
                for j in cam_indices:
                    batch_indices.append(same_pid_indices[j])
            else:
                select_indices = no_index(same_pid_indices, pid_index)
                if not select_indices:
                    # only one image for this identity
                    ind_indices = [0] * (self.num_instances - 1)
                elif len(select_indices) >= self.num_instances:
                    ind_indices = np.random.choice(
                        select_indices,
                        size=self.num_instances - 1,
                        replace=False,
                    )
                else:
                    ind_indices = np.random.choice(
                        select_indices,
                        size=self.num_instances - 1,
                        replace=True,
                    )
                for j in ind_indices:
                    batch_indices.append(same_pid_indices[j])

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
        seed=0,
        shuffle=True,
        batch_size=32,
        num_instances=4,
        round_up=True,
    ):
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

        # avoid having to run pipelines
        self.data_infos = copy.deepcopy(
            [info["sampler_info"] for info in dataset.data_infos]
        )

        self.index_pid = dict()
        self.pid_index = defaultdict(list)
        self.pid_cam = defaultdict(list)
        for index, info in enumerate(self.data_infos):
            pid = info["pid"]
            camid = info["camid"]
            self.index_pid[index] = pid
            self.pid_index[pid].append(index)
            self.pid_cam[pid].append(camid)

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
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            identities = torch.randperm(
                self.num_identities, generator=g
            ).tolist()
        else:
            identities = torch.arange(self.num_identities).tolist()

        tot = self.num_iterations * self.num_pids_per_batch
        if self.round_up and len(identities) % self.num_pids_per_batch != 0:
            # pad
            identities = (identities * int(tot / len(identities) + 1))[:tot]
        elif len(identities) % self.num_pids_per_batch != 0:
            # drop
            identities = identities[
                : -(len(identities) % self.num_pids_per_batch)
            ]

        indices = []
        for i in identities:
            batch_indices = []

            pid_index = np.random.choice(self.pid_index[self.pids[i]])
            batch_indices.append(pid_index)
            data = self.data_infos[pid_index]
            pid = data["pid"]
            camid = data["camid"]

            same_pid_indices = self.pid_index[pid]
            cam_list = self.pid_cam[pid]

            select_cams = no_index(cam_list, camid)

            if select_cams:
                if len(select_cams) >= self.num_instances:
                    cam_indices = np.random.choice(
                        select_cams,
                        size=self.num_instances - 1,
                        replace=False,
                    )
                else:
                    cam_indices = np.random.choice(
                        select_cams,
                        size=self.num_instances - 1,
                        replace=True,
                    )
                for j in cam_indices:
                    batch_indices.append(same_pid_indices[j])
            else:
                select_indices = no_index(same_pid_indices, pid_index)
                if not select_indices:
                    # only one image for this identity
                    ind_indices = [0] * (self.num_instances - 1)
                elif len(select_indices) >= self.num_instances:
                    ind_indices = np.random.choice(
                        select_indices,
                        size=self.num_instances - 1,
                        replace=False,
                    )
                else:
                    ind_indices = np.random.choice(
                        select_indices,
                        size=self.num_instances - 1,
                        replace=True,
                    )
                for j in ind_indices:
                    batch_indices.append(same_pid_indices[j])

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
