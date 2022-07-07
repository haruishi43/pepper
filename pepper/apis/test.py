#!/usr/bin/env python3

import os.path as osp
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmcv.engine import collect_results_cpu, collect_results_gpu


def single_gpu_test(
    model,
    data_loader,
):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        batch_size = len(result)
        results.extend(result)

        batch_size = data["img"].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    return results


def multi_gpu_test(
    model,
    data_loader,
    tmpdir=None,
    gpu_collect=False,
):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        # Check if tmpdir is valid for cpu_collect
        if (not gpu_collect) and (tmpdir is not None and osp.exists(tmpdir)):
            raise OSError(
                (
                    f"The tmpdir {tmpdir} already exists.",
                    " Since tmpdir will be deleted after testing,",
                    " please make sure you specify an empty one.",
                )
            )
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)
    dist.barrier()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

        if rank == 0:
            batch_size = data["img"].size(0)

            # FIXME: seems like validation set is not running distributed
            # number of validation dataset is len(data_loader) * 2

            for _ in range(batch_size * world_size):
                prog_bar.update()

            # for _ in range(batch_size):
            #     prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results
