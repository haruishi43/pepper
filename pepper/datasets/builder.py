#!/usr/bin/env python3

import copy
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg

DATASETS = Registry("dataset")
PIPELINES = Registry("pipeline")
SAMPLERS = Registry("sampler")


def build_dataset(cfg, default_args=None):
    from .dataset_wrappers import (
        ClassBalancedDataset,
        ConcatDataset,
        KFoldDataset,
        RepeatDataset,
    )

    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg["type"] == "ConcatDataset":
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg["datasets"]],
            separate_eval=cfg.get("separate_eval", True),
        )
    elif cfg["type"] == "RepeatDataset":
        dataset = RepeatDataset(
            build_dataset(cfg["dataset"], default_args), cfg["times"]
        )
    elif cfg["type"] == "ClassBalancedDataset":
        dataset = ClassBalancedDataset(
            build_dataset(cfg["dataset"], default_args), cfg["oversample_thr"]
        )
    elif cfg["type"] == "KFoldDataset":
        cp_cfg = copy.deepcopy(cfg)
        if cp_cfg.get("test_mode", None) is None:
            cp_cfg["test_mode"] = (default_args or {}).pop("test_mode", False)
        cp_cfg["dataset"] = build_dataset(cp_cfg["dataset"], default_args)
        cp_cfg.pop("type")
        dataset = KFoldDataset(**cp_cfg)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(
    dataset,
    samples_per_gpu,
    workers_per_gpu,
    num_gpus=1,
    dist=True,
    shuffle=True,
    round_up=True,
    seed=None,
    pin_memory=True,
    persistent_workers=True,
    sampler_cfg=None,
):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        round_up (bool): Whether to round up the length of dataset by adding
            extra samples to make it evenly divisible. Default: True.
        pin_memory (bool): Whether to use pin_memory in DataLoader.
            Default: True
        persistent_workers (bool): If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers Dataset instances alive.
            The argument also has effect in PyTorch>=1.7.0.
            Default: True
        sampler_cfg (dict): sampler configuration to override the default
            sampler
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()

    if dist:
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    # setup sampler
    # NOTE: we don't use sampler when we run validation or test
    # custom sampler logic
    if sampler_cfg:
        # overwrite
        sampler_cfg.update(shuffle=shuffle)
        if seed is not None:
            sampler_cfg.update(seed=seed)

        # some sampler-specific arguments that needs to be overwriten before
        if sampler_cfg.get('batch_size', None):
            sampler_cfg.update(batch_size=batch_size)

        sampler = build_sampler(
            sampler_cfg,
            default_args=dict(
                dataset=dataset,
                num_replicas=world_size,
                rank=rank,
            )
        )
    elif dist:
        sampler_cfg = dict(
            type='BalancedDistributedSampler',
            batch_size=batch_size,
            num_instances=batch_size // 8,
            shuffle=shuffle,
            round_up=round_up,
            seed=seed if seed is not None else 0,
        )
        default_args = dict(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
        )
        sampler = build_sampler(
            sampler_cfg,
            default_args=default_args,
        )
    else:
        sampler = None

    # If sampler exists, turn off dataloader shuffle
    if sampler is not None:
        shuffle = False

    init_fn = (
        partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)
        if seed is not None
        else None
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        persistent_workers=persistent_workers,
    )

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_sampler(cfg, default_args=None):
    if cfg is None:
        return None
    else:
        return build_from_cfg(cfg, SAMPLERS, default_args=default_args)
