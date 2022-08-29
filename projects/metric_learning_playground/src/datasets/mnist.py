#!/usr/bin/env python3

import codecs
import os.path as osp

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

from mmcls.datasets.builder import DATASETS
from mmcls.datasets.mnist import MNIST
from mmcls.core.evaluation import precision_recall_f1, support
from mmcls.models.losses import accuracy
from mmcls.datasets.utils import rm_suffix


@DATASETS.register_module()
class CustomMNIST(MNIST):

    def load_annotations(self):
        train_image_file = osp.join(
            self.data_prefix, rm_suffix(self.resources['train_image_file'][0]))
        train_label_file = osp.join(
            self.data_prefix, rm_suffix(self.resources['train_label_file'][0]))
        test_image_file = osp.join(
            self.data_prefix, rm_suffix(self.resources['test_image_file'][0]))
        test_label_file = osp.join(
            self.data_prefix, rm_suffix(self.resources['test_label_file'][0]))

        if not osp.exists(train_image_file) or not osp.exists(
                train_label_file) or not osp.exists(
                    test_image_file) or not osp.exists(test_label_file):
            self.download()

        _, world_size = get_dist_info()
        if world_size > 1:
            dist.barrier()
            assert osp.exists(train_image_file) and osp.exists(
                train_label_file) and osp.exists(
                    test_image_file) and osp.exists(test_label_file), \
                'Shared storage seems unavailable. Please download dataset ' \
                f'manually through {self.resource_prefix}.'

        train_set = (read_image_file(train_image_file),
                     read_label_file(train_label_file))
        test_set = (read_image_file(test_image_file),
                    read_label_file(test_label_file))

        if not self.test_mode:
            imgs, gt_labels = train_set
        else:
            imgs, gt_labels = test_set

        data_infos = []
        for img, gt_label in zip(imgs, gt_labels):
            gt_label = np.array(gt_label, dtype=np.int64)
            info = {'img': img.numpy(), 'gt_label': gt_label}
            data_infos.append(info)
        return data_infos

    def evaluate(
        self,
        results,
        metric="accuracy",
        metric_options=None,
        indices=None,
        logger=None,
    ):
        """Evaluate the dataset.
        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            indices (list, optional): The indices of samples corresponding to
                the results. Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {"topk": (1, 5)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "support",
        ]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        if indices is not None:
            gt_labels = gt_labels[indices]
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, (
            "dataset testing results should "
            "be of the same length as gt_labels."
        )

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f"metric {invalid_metrics} is not supported.")

        topk = metric_options.get("topk", (1, 5))
        thrs = metric_options.get("thrs")
        average_mode = metric_options.get("average_mode", "macro")

        if "accuracy" in metrics:
            if thrs is not None:
                acc = accuracy(results, gt_labels, topk=topk, thrs=thrs)
            else:
                acc = accuracy(results, gt_labels, topk=topk)
            if isinstance(topk, tuple):
                eval_results_ = {
                    f"accuracy_top-{k}": a for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {"accuracy": acc}
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update(
                        {
                            f"{key}_thr_{thr:.2f}": value.item()
                            for thr, value in zip(thrs, values)
                        }
                    )
            else:
                eval_results.update(
                    {k: v.item() for k, v in eval_results_.items()}
                )

        if "support" in metrics:
            support_value = support(
                results, gt_labels, average_mode=average_mode
            )
            eval_results["support"] = support_value

        precision_recall_f1_keys = ["precision", "recall", "f1_score"]
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            if thrs is not None:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode, thrs=thrs
                )
            else:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode
                )
            for key, values in zip(
                precision_recall_f1_keys, precision_recall_f1_values
            ):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_results.update(
                            {
                                f"{key}_thr_{thr:.2f}": value
                                for thr, value in zip(thrs, values)
                            }
                        )
                    else:
                        eval_results[key] = values

        # TODO: add visualizations

        return eval_results


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def open_maybe_compressed_file(path):
    """Return a file object that possibly decompresses 'path' on the fly.
    Decompression occurs when argument `path` is a string and ends with '.gz'
    or '.xz'.
    """
    if not isinstance(path, str):
        return path
    if path.endswith('.gz'):
        import gzip
        return gzip.open(path, 'rb')
    if path.endswith('.xz'):
        import lzma
        return lzma.open(path, 'rb')
    return open(path, 'rb')


def read_sn3_pascalvincent_tensor(path, strict=True):
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-
    io.lsh').
    Argument may be a filename, compressed filename, or file object.
    """
    # typemap
    if not hasattr(read_sn3_pascalvincent_tensor, 'typemap'):
        read_sn3_pascalvincent_tensor.typemap = {
            8: (torch.uint8, np.uint8, np.uint8),
            9: (torch.int8, np.int8, np.int8),
            11: (torch.int16, np.dtype('>i2'), 'i2'),
            12: (torch.int32, np.dtype('>i4'), 'i4'),
            13: (torch.float32, np.dtype('>f4'), 'f4'),
            14: (torch.float64, np.dtype('>f8'), 'f8')
        }
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert nd >= 1 and nd <= 3
    assert ty >= 8 and ty <= 14
    m = read_sn3_pascalvincent_tensor.typemap[ty]
    s = [get_int(data[4 * (i + 1):4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    # return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)
    # FIXED unwritable tensor
    return torch.from_numpy(parsed.astype(m[2])).view(*s)


def read_label_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert (x.dtype == torch.uint8)
    assert (x.ndimension() == 1)
    return x.long()


def read_image_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert (x.dtype == torch.uint8)
    assert (x.ndimension() == 3)
    return x
