#!/usr/bin/env python3

import argparse
import copy
import os
import os.path as osp
import time

import torch

import mmcv
from mmcv import Config, DictAction
from mmcv.runner import init_dist

from pepper.apis import init_random_seed, set_random_seed, train_model
from pepper.models import build_reid
from pepper.datasets import build_dataset
from pepper.utils import collect_env, get_root_logger, setup_multi_processes


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

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    # overwrite model loading directory if provided in the args
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
        # assume that we are using DataParallel
        # use all devices in CUDA_VISIBLE_DEVICES=
        assert torch.cuda.is_available(), "ERR: no CUDA devices"
        cfg.gpu_ids = [i for i in range(torch.cuda.device_count())]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()

    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    meta["env_info"] = env_info

    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)

    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds. Force setting fixed seed and deterministic=True in SOT
    # configs
    if args.seed is not None:
        cfg.seed = args.seed
    elif cfg.get("seed", None) is None:
        cfg.seed = init_random_seed()

    deterministic = (
        True if args.deterministic else cfg.get("deterministic", False)
    )

    set_random_seed(cfg.seed, deterministic=deterministic)
    meta["seed"] = cfg.seed
    logger.info(
        f"Set random seed to {cfg.seed}, " f"deterministic: {deterministic}"
    )

    if cfg.get("train_cfg", False):
        model = build_reid(
            cfg.model,
            train_cfg=cfg.train_cfg,
            test_cfg=cfg.test_cfg,
        )
    else:
        model = build_reid(cfg.model)
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmtrack version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            config=cfg.pretty_text,
            NUM_PIDS=datasets[0].NUM_PIDS,
        )
    # add an attribute for visualization convenience
    model.NUM_PIDS = datasets[0].NUM_PIDS
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta,
    )


if __name__ == "__main__":
    main()
