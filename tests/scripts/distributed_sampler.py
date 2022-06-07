#!/usr/bin/env python3

import argparse
import os
import os.path as osp
import time
import warnings

import torch
import torch.distributed as dist

from mmcv import Config, DictAction
from mmcv.runner import init_dist

from pepper.apis import init_random_seed, set_random_seed
from pepper.datasets import build_dataset, build_dataloader
from pepper.utils import collect_env, get_root_logger, setup_multi_processes


def iterate_dataset(
    dataset,
    cfg,
    distributed=True,
    timestamp=None,
    meta=None,
):
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    sampler_cfg = cfg.data.get("sampler", None)

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # `num_gpus` will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            shuffle=True,  # NOTE: debug shuffle (overwrite cfg)
            round_up=True,
            seed=cfg.seed,
            sampler_cfg=sampler_cfg,
        )
        for ds in dataset
    ]

    loader = data_loaders[0]

    print("iterating...")
    dist.barrier()

    for i, data in enumerate(loader):
        meta = data["img_metas"].data[0]
        # print(meta)
        camids = [m["camid"] for m in meta]
        debug_idx = [m["debug_index"] for m in meta]

        # FIXME: need help flushing
        # logger.info(f">>> {i}: {debug_idx}")  # logger only logs in rank 0
        print(f">>> {i}: index {debug_idx}")
        dist.barrier()
        print(f">>> {i}: ids {data['gt_label'].data}")
        dist.barrier()
        print(f">>> {i}: camids {camids}")
        dist.barrier()

    dist.barrier()
    for i, data in enumerate(loader):
        meta = data["img_metas"].data[0]
        # print(meta)
        camids = [m["camid"] for m in meta]
        debug_idx = [m["debug_index"] for m in meta]

        # FIXME: need help flushing
        # logger.info(f">>> {i}: {debug_idx}")  # logger only logs in rank 0
        print(f">>> {i}: index {debug_idx}")
        dist.barrier()
        print(f">>> {i}: ids {data['gt_label'].data}")
        dist.barrier()
        print(f">>> {i}: camids {camids}")
        dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--resume-from", help="the checkpoint file to resume from"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="whether not to evaluate the checkpoint during training",
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--gpus",
        type=int,
        help="(Deprecated, please use --gpu-id) number of gpus to use "
        "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="(Deprecated, please use --gpu-id) ids of gpus to use "
        "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="id of gpu to use "
        "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    # Setup for torch.dist
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn(
            "`--gpus` is deprecated because we only support "
            "single GPU mode in non-distributed training. "
            "Use `gpus=1` now."
        )
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn(
            "`--gpu-ids` is deprecated, please use `--gpu-id`. "
            "Because we only support single GPU mode in "
            "non-distributed training. Use the first GPU "
            "in `gpu_ids` now."
        )
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # FIXME: force distributed
    assert distributed

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join("tests/logs/", f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    meta["env_info"] = env_info

    # log some basic info
    # logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    # logger.info(f"Distributed training: {distributed}")
    # logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds. Force setting fixed seed and deterministic=True in SOT
    # configs
    if args.seed is not None:
        cfg.seed = args.seed
    elif cfg.get("seed", None) is None:
        cfg.seed = init_random_seed()

    deterministic = (
        True if args.deterministic else cfg.get("deterministic", False)
    )
    logger.info(
        f"Set random seed to {cfg.seed}, " f"deterministic: {deterministic}"
    )
    set_random_seed(cfg.seed, deterministic=deterministic)
    meta["seed"] = cfg.seed

    datasets = [build_dataset(cfg.data.train)]

    iterate_dataset(
        datasets,
        cfg,
        distributed=distributed,
        timestamp=timestamp,
        meta=meta,
    )


if __name__ == "__main__":
    main()
