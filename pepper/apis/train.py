#!/usr/bin/env python3

import random
import warnings

import numpy as np
import torch
import torch.distributed as dist

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    DistSamplerSeedHook,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
    get_dist_info,
)

from pepper.datasets import build_dataset, replace_ImageToTensor
from pepper.utils import find_latest_checkpoint

from pepper.core import DistEvalHook, EvalHook
from pepper.datasets import build_dataloader
from pepper.utils import get_root_logger


def init_random_seed(seed=None, device="cuda"):
    """Initialize random seed.
    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
    meta=None,
):
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    sampler_cfg = cfg.get("sampler", None)

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            dist=distributed,
            round_up=True,
            seed=cfg.seed,
            sampler_cfg=sampler_cfg,
        )
        for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    # TODO: add center loss optimizer
    # TODO: OptimizerHook doesn't support multiple losses
    # instead we should directly optimize in the `train_step`
    # cfg.optimizer_cfg = None  # this should make it so that OptimizerHook won't initialize

    if cfg.get("runner") is None:
        cfg.runner = {
            "type": "EpochBasedRunner",
            "max_epochs": cfg.total_epochs,
        }
        warnings.warn(
            "config is now expected to have a `runner` section, "
            "place set `runner` in your config.",
            UserWarning,
        )
    else:
        if "total_epochs" in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,  # NOTE: deprecated
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta,
        ),
    )

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed
        )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
        custom_hooks_config=cfg.get("custom_hooks", None),
    )
    if distributed and cfg.runner["type"] == "EpochBasedRunner":
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        if isinstance(cfg.data.val, (list, tuple)):
            for val_cfg in cfg.data.val:
                # Support batch_size > 1 in validation
                val_samples_per_gpu = val_cfg.pop("samples_per_gpu", 1)
                if val_samples_per_gpu > 1:
                    # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                    val_cfg.pipeline = replace_ImageToTensor(val_cfg.pipeline)

                val_dataset = build_dataset(val_cfg, dict(eval_mode=True))

                val_dataloader = build_dataloader(
                    val_dataset,
                    samples_per_gpu=val_samples_per_gpu,
                    workers_per_gpu=cfg.data.workers_per_gpu,
                    dist=distributed,
                    shuffle=False,
                    round_up=True,
                    is_val=True,
                )
                eval_cfg = cfg.get("evaluation", {})
                eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
                eval_hook = DistEvalHook if distributed else EvalHook
                # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
                # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
                runner.register_hook(
                    eval_hook(val_dataloader, **eval_cfg), priority="LOW"
                )

        else:
            # Support batch_size > 1 in validation
            val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
            if val_samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.val.pipeline = replace_ImageToTensor(
                    cfg.data.val.pipeline
                )

            val_dataset = build_dataset(cfg.data.val, dict(eval_mode=True))

            val_dataloader = build_dataloader(
                val_dataset,
                samples_per_gpu=val_samples_per_gpu,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False,
                round_up=True,
                is_val=True,
            )
            eval_cfg = cfg.get("evaluation", {})
            eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
            eval_hook = DistEvalHook if distributed else EvalHook
            # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
            # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
            runner.register_hook(
                eval_hook(val_dataloader, **eval_cfg), priority="LOW"
            )

    resume_from = None
    if cfg.resume_from is None and cfg.get("auto_resume"):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
