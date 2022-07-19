#!/usr/bin/env python3

"""
Iter-based sampler
"""

import copy
import itertools
from collections import defaultdict

import numpy as np

import torch.distributed as dist
from torch.utils.data.sampler import Sampler

from ..builder import SAMPLERS
from .utils import no_index, reorder_index


class BaseInfiniteDistributedSampler(Sampler):
    """
    Warning! This is not your average DistributedSampler

    Technically, it can be used in non-distributed samplers...
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available"
                )
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available"
                )
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )

        self.rank = rank
        self.num_replicas = num_replicas

        # avoid having to run pipelines
        data_infos = copy.deepcopy(
            [info["img_info"] for info in dataset.data_infos]
        )
        self.data_infos = data_infos

    def __len__(self):
        # NOTE: this is needed but not accurate
        return len(self.data_infos) // self.num_replicas

    def __iter__(self):
        start = self.rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self.num_replicas
        )

    def _infinite_indices(self):
        pass


@SAMPLERS.register_module()
class InfiniteBalancedIdentityDistributedSampler(
    BaseInfiniteDistributedSampler
):
    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        batch_size: int = 32,
        num_instances: int = 4,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank)

        self.batch_size = batch_size * self.num_replicas
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances

        self.index_pid = dict()
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)

        for index, info in enumerate(self.data_infos):
            pid = info["pid"]
            camid = info["camid"]
            self.index_pid[index] = pid
            self.pid_cam[pid].append(camid)
            self.pid_index[pid].append(index)

        self.pids = sorted(list(self.pid_index.keys()))
        self.num_identities = len(self.pids)

        self.seed = int(seed)

    def _infinite_indices(self):
        np.random.seed(self.seed)
        while True:
            # Shuffle identity list
            identities = np.random.permutation(self.num_identities)

            # If remaining identities cannot be enough for a batch,
            # just drop the remaining parts
            drop_indices = self.num_identities % (
                self.num_pids_per_batch * self.num_replicas
            )
            if drop_indices:
                identities = identities[:-drop_indices]

            batch_indices = []
            for kid in identities:
                i = np.random.choice(self.pid_index[self.pids[kid]])
                i_cam = self.data_infos[i]["camid"]
                batch_indices.append(i)
                pid_i = self.index_pid[i]
                cams = self.pid_cam[pid_i]
                index = self.pid_index[pid_i]
                select_cams = no_index(cams, i_cam)

                if select_cams:
                    if len(select_cams) >= self.num_instances:
                        cam_indexes = np.random.choice(
                            select_cams,
                            size=self.num_instances - 1,
                            replace=False,
                        )
                    else:
                        cam_indexes = np.random.choice(
                            select_cams,
                            size=self.num_instances - 1,
                            replace=True,
                        )
                    for kk in cam_indexes:
                        batch_indices.append(index[kk])
                else:
                    select_indexes = no_index(index, i)
                    if not select_indexes:
                        # Only one image for this identity
                        ind_indexes = [0] * (self.num_instances - 1)
                    elif len(select_indexes) >= self.num_instances:
                        ind_indexes = np.random.choice(
                            select_indexes,
                            size=self.num_instances - 1,
                            replace=False,
                        )
                    else:
                        ind_indexes = np.random.choice(
                            select_indexes,
                            size=self.num_instances - 1,
                            replace=True,
                        )

                    for kk in ind_indexes:
                        batch_indices.append(index[kk])

                if len(batch_indices) == self.batch_size:
                    yield from reorder_index(batch_indices, self.num_replicas)
                    batch_indices = []


@SAMPLERS.register_module()
class InfiniteSetReWeightDistributedSampler(BaseInfiniteDistributedSampler):
    def __init__(
        self,
        dataset,
        set_weight: list,
        num_replicas=None,
        rank=None,
        batch_size: int = 32,
        num_instances: int = 4,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank)

        self.batch_size = batch_size * self.num_replicas
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances

        self.set_weight = set_weight

        assert (
            self.batch_size % (sum(self.set_weight) * self.num_instances) == 0
            and self.batch_size > sum(self.set_weight) * self.num_instances
        ), "Batch size must be divisible by the sum set weight"

        self.index_pid = dict()
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)

        self.cam_pid = defaultdict(list)

        for index, info in enumerate(self.data_infos):
            pid = info["pid"]
            camid = info["camid"]
            self.index_pid[index] = pid
            self.pid_cam[pid].append(camid)
            self.pid_index[pid].append(index)
            self.cam_pid[camid].append(pid)

        # Get sampler prob for each cam
        self.set_pid_prob = defaultdict(list)
        for camid, pid_list in self.cam_pid.items():
            index_per_pid = []
            for pid in pid_list:
                index_per_pid.append(len(self.pid_index[pid]))
            cam_image_number = sum(index_per_pid)
            prob = [i / cam_image_number for i in index_per_pid]
            self.set_pid_prob[camid] = prob

        self.pids = sorted(list(self.pid_index.keys()))
        self.num_identities = len(self.pids)

        self.seed = int(seed)

    def _infinite_indices(self):
        np.random.seed(self.seed)
        while True:
            batch_indices = []
            for camid in range(len(self.cam_pid.keys())):
                select_pids = np.random.choice(
                    self.cam_pid[camid],
                    size=self.set_weight[camid],
                    replace=False,
                    p=self.set_pid_prob[camid],
                )
                for pid in select_pids:
                    index_list = self.pid_index[pid]
                    if len(index_list) > self.num_instances:
                        select_indexs = np.random.choice(
                            index_list, size=self.num_instances, replace=False
                        )
                    else:
                        select_indexs = np.random.choice(
                            index_list, size=self.num_instances, replace=True
                        )

                    batch_indices += select_indexs
            np.random.shuffle(batch_indices)

            if len(batch_indices) == self.batch_size:
                yield from reorder_index(batch_indices, self.num_replicas)


@SAMPLERS.register_module()
class InfiniteNaiveIdentityDistributedSampler(BaseInfiniteDistributedSampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        batch_size: int = 32,
        num_instances: int = 4,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank)

        self.batch_size = batch_size * self.num_replicas
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances

        self.pid_index = defaultdict(list)

        for index, info in enumerate(self.data_infos):
            pid = info["pid"]
            self.pid_index[pid].append(index)

        self.pids = sorted(list(self.pid_index.keys()))
        self.num_identities = len(self.pids)

        self.seed = int(seed)

    def _infinite_indices(self):
        np.random.seed(self.seed)
        while True:
            avl_pids = copy.deepcopy(self.pids)
            batch_idxs_dict = {}

            batch_indices = []
            while len(avl_pids) >= self.num_pids_per_batch:
                selected_pids = np.random.choice(
                    avl_pids, self.num_pids_per_batch, replace=False
                ).tolist()
                for pid in selected_pids:
                    # Register pid in batch_idxs_dict if not
                    if pid not in batch_idxs_dict:
                        idxs = copy.deepcopy(self.pid_index[pid])
                        if len(idxs) < self.num_instances:
                            idxs = np.random.choice(
                                idxs, size=self.num_instances, replace=True
                            ).tolist()
                        np.random.shuffle(idxs)
                        batch_idxs_dict[pid] = idxs

                    avl_idxs = batch_idxs_dict[pid]
                    for _ in range(self.num_instances):
                        batch_indices.append(avl_idxs.pop())

                    if len(avl_idxs) < self.num_instances:
                        avl_pids.remove(pid)

                if len(batch_indices) == self.batch_size:
                    yield from reorder_index(batch_indices, self.num_replicas)
                    batch_indices = []
