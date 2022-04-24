#!/usr/bin/env python3

import copy
import os.path as osp
from abc import ABCMeta, abstractmethod
from os import PathLike

import numpy as np
import torch
from torch.utils.data import Dataset

from .pipelines import Compose


def expanduser(path):
    if isinstance(path, (str, PathLike)):
        return osp.expanduser(path)
    else:
        return path


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base dataset.
    Args:
        data_prefix (str): the prefix of data path
        pipeline (list): a list of dict, where each element represents
            a operation defined in `pepper.datasets.pipelines`
        ann_file (str | None): the annotation file. When ann_file is str,
            the subclass is expected to read from the ann_file. When ann_file
            is None, the subclass is expected to read according to data_prefix
        test_mode (bool): in train mode or test mode
    """

    NUM_PIDS = None
    EVAL_KEYS = ("query", "gallery")

    # eval mode
    _is_eval = False

    # number of query images (to index `data_infos`)
    _num_query = None  # HACK: using this for indexing back query and gallery
    _num_gallery = None

    def __init__(
        self,
        data_prefix,
        pipeline,
        ann_file=None,
        eval_mode=False,
    ):
        super(BaseDataset, self).__init__()

        # NOTE: since evaluation dataset should be handled in this class as well,
        # we use `dict` for `data_prefix` and `ann_files` during val or test set.
        # The keys should be `query` and `gallery`

        if eval_mode:
            assert isinstance(data_prefix, dict) and isinstance(
                ann_file, dict
            ), "for validation, `data_prefix` and `ann_file` must be dict."

            _data_prefix = dict()
            for key in data_prefix.keys():
                assert key in self.EVAL_KEYS
                _data_prefix[key] = expanduser(data_prefix[key])
            self.data_prefix = _data_prefix

            _ann_file = dict()
            for key in ann_file.keys():
                assert key in self.EVAL_KEYS
                _ann_file[key] = expanduser(ann_file[key])
            self.ann_file = _ann_file
        else:
            # does not check in case of `None` since we would like to run data
            # that doesn't have annotations
            self.data_prefix = expanduser(data_prefix)
            self.ann_file = expanduser(ann_file)

        self._is_eval = eval_mode

        # data_infos is a List of dicts
        # the dicts should contain 'sampler_info' and 'img_info'
        self.data_infos = self.load_annotations()

        # setup pipeline
        self.pipeline = Compose(pipeline)

    @abstractmethod
    def load_annotations(self):
        pass

    def get_query_infos(self):
        assert self._is_eval
        return self.data_infos[: self._num_query]

    def get_gallery_infos(self):
        assert self._is_eval
        return self.data_infos[self._num_query :]

    def prepare_data(self, data):
        return self.pipeline(data)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.data_infos[idx])
        return self.prepare_data(data)

    def get_gt_labels(self):
        gt_labels = np.array([data["gt_label"] for data in self.data_infos])
        return gt_labels

    def get_pids(self):
        infos = copy.deepcopy(self.data_infos)
        return np.asarray([info["img_info"]["pid"] for info in infos])

    def get_camids(self):
        infos = copy.deepcopy(self.data_infos)
        return np.asarray([info["img_info"]["camid"] for info in infos])

    def evaluate(
        self,
        results,
        metric="mAP",
        metric_options=None,
        logger=None,
    ):
        """Evaluate the ReID dataset

        - results: dict

        """
        assert self._is_eval, "ERR: not in eval mode"
        # NOTE: we assume that the results are in order [query, gallery]

        if metric_options is None:
            metric_options = dict(rank_list=[1, 5, 10, 25], max_rank=20)
        for rank in metric_options["rank_list"]:
            assert rank >= 1 and rank <= metric_options["max_rank"]
        if isinstance(metric, list):
            metric = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError("metric must be a list or a str")

        allowed_metrics = ["mAP", "CMC"]
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f"metric {metric} is not supported.")

        # assert that results is list of tensors
        results = [result.data.cpu() for result in results]
        features = torch.stack(results)
        pids = torch.from_numpy(self.get_pids())
        camids = torch.from_numpy(self.get_camids())

        # separate query and gallery
        assert len(features) == len(self.data_infos)
        q_feat = features[: self._num_query]
        g_feat = features[self._num_query :]
        q_pids = pids[: self._num_query]
        g_pids = pids[self._num_query :]
        q_camids = camids[: self._num_query]
        g_camids = camids[self._num_query :]
