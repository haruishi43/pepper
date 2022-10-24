#!/usr/bin/env python3

import argparse
import os
from numbers import Number

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)

from pepper.apis import multi_gpu_test, single_gpu_test
from pepper.datasets import build_dataloader, build_dataset
from pepper.models import build_reid
from pepper.utils import get_root_logger, setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(description="Test a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")

    # TODO: add output function later
    # parser.add_argument("--out", help="output result file")
    # out_options = ["class_scores", "pred_score", "pred_label", "pred_class"]
    # parser.add_argument(
    #     "--out-items",
    #     nargs="+",
    #     default=["all"],
    #     choices=out_options + ["none", "all"],
    #     help="Besides metrics, what items will be included in the output "
    #     f'result file. You can choose some of ({", ".join(out_options)}), '
    #     'or use "all" to include all above, or use "none" to disable all of '
    #     "above. Defaults to output all.",
    #     metavar="",
    # )

    # TODO: add visualization
    # parser.add_argument("--show", action="store_true", help="show results")
    # parser.add_argument(
    #     "--show-dir", help="directory where painted images will be saved"
    # )

    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results",
    )
    parser.add_argument("--tmpdir", help="tmp dir for writing some results")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--metric-options",
        nargs="+",
        action=DictAction,
        default={},
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be parsed as a dict metric_options for dataset.evaluate()"
        " function.",
    )
    parser.add_argument(
        "--show-options",
        nargs="+",
        action=DictAction,
        help="custom options for show_result. key-value pair in xxx=yyy."
        "Check available options in `model.show_result`.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="id of gpu to use " "(only applicable to non-distributed testing)",
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

    # assert (
    #     args.metrics or args.out
    # ), "Please specify at least one of output path and evaluation metrics."

    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader

    if cfg.data.samples_per_gpu > 1:
        # for now, force single batch
        cfg.data.samples_per_gpu = 1
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        # from pepper.datasets import replace_ImageToTensor
        # cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test, default_args=dict(eval_mode=True))
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=True,
        is_val=True,
    )

    # build the model and load checkpoint
    model = build_reid(cfg.model)
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    # FIXME: NUM_PIDS might differ between train and test sets
    if "NUM_PIDS" in checkpoint.get("meta", {}):
        NUM_PIDS = checkpoint["meta"]["NUM_PIDS"]
    else:
        NUM_PIDS = dataset.NUM_PIDS

    model.head.num_classes = NUM_PIDS

    if not distributed:
        # SyncBN is not supported for DP
        model = revert_sync_batchnorm(model)

        if args.device == "cpu":
            model = model.cpu()
        else:
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)

        # TODO: add visualization with `show`
        show_kwargs = {} if args.show_options is None else args.show_options
        outputs = single_gpu_test(
            model, data_loader, args.show, args.show_dir, **show_kwargs
        )
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect
        )

    # do something with the outputs
    # TODO: make this argparse
    arg_metric = ["metric", "mAP", "CMC", "mINP"]
    arg_metric_options = None
    arg_use_metric_cuhk03 = False

    rank, _ = get_dist_info()
    if rank == 0:
        results = {}
        logger = get_root_logger()
        eval_results = dataset.evaluate(
            results=outputs,
            metric=arg_metric,
            metric_options=arg_metric_options,
            use_metric_cuhk03=arg_use_metric_cuhk03,
            logger=logger,
        )
        results.update(eval_results)
        for k, v in eval_results.items():
            if isinstance(v, np.ndarray):
                v = [round(out, 2) for out in v.tolist()]
            elif isinstance(v, Number):
                v = round(v, 2)
            else:
                raise ValueError(f"Unsupport metric type: {type(v)}")
            print(f"{k} : {v}")

        # TODO: output to file
        # if args.out:
        #     if "none" not in args.out_items:
        #         scores = np.vstack(outputs)
        #         pred_score = np.max(scores, axis=1)
        #         pred_label = np.argmax(scores, axis=1)
        #         pred_class = [NUM_PIDS[lb] for lb in pred_label]
        #         res_items = {
        #             "class_scores": scores,
        #             "pred_score": pred_score,
        #             "pred_label": pred_label,
        #             "pred_class": pred_class,
        #         }
        #         if "all" in args.out_items:
        #             results.update(res_items)
        #         else:
        #             for key in args.out_items:
        #                 results[key] = res_items[key]
        #     print(f"\ndumping results to {args.out}")
        #     mmcv.dump(results, args.out)


if __name__ == "__main__":
    main()
