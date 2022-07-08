#!/usr/bin/env python3

import copy
import os.path as osp
import warnings
from abc import ABCMeta, abstractmethod
from os import PathLike

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from pepper.core.evaluation import evaluate

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
    """

    NUM_PIDS = None
    NUM_CAMIDS = None
    EVAL_KEYS = ("query", "gallery")

    # eval mode  (default: false)
    _is_eval = False

    # check if data is query and gallery (default: false)
    _is_query_gallery = False

    # number of query images (to index `data_infos`)
    _num_query = None  # HACK: using this for indexing back query and gallery
    _num_gallery = None  # not really used (only for validation)

    def __init__(
        self,
        data_prefix,
        pipeline,
        ann_file=None,
        eval_mode=False,
        num_pids=None,
        num_camids=None,
    ):
        super(BaseDataset, self).__init__()

        # NOTE: since evaluation dataset should be handled in this class as well,
        # we use `dict` for `data_prefix` and `ann_files` during val or test set.
        # The keys should be `query` and `gallery`

        if isinstance(data_prefix, dict):
            # we assume we're loading query/gallery pairs
            assert isinstance(
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

            self._is_query_gallery = True
            self._is_eval = eval_mode

            if not eval_mode:
                warnings.warn("Using query/gallery as training data")

        else:
            # we assume we're loading train split
            assert not eval_mode, "training split cannot be used for evaluation"

            # does not check in case of `None` since we would like to run data
            # that doesn't have annotations
            self.data_prefix = expanduser(data_prefix)
            self.ann_file = expanduser(ann_file)

        self._is_eval = eval_mode

        # data_infos is a List of dicts
        # the dicts should contain 'sampler_info' and 'img_info'
        self.data_infos = self.load_annotations()

        # add globals
        if num_pids is None:
            self.NUM_PIDS = len(np.unique(self.get_pids()))
        else:
            self.NUM_PIDS = num_pids

        if num_camids is None:
            self.NUM_CAMIDS = len(np.unique(self.get_camids()))
        else:
            self.NUM_CAMIDS = num_camids

        # setup pipeline
        self.pipeline = Compose(pipeline)

    @abstractmethod
    def load_annotations(self):
        pass

    def get_query_infos(self):
        assert self._is_query_gallery and self._is_eval
        return self.data_infos[: self._num_query]

    def get_gallery_infos(self):
        assert self._is_query_gallery and self._is_eval
        return self.data_infos[self._num_query :]

    def prepare_data(self, data):
        # NOTE: might need to prepare the sample before handing it to pipeline
        return self.pipeline(data)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):

        # NOTE: some functions in the pipeline may be inplace
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
        metric=["metric", "mAP", "CMC"],
        metric_options=None,
        normalize_features=False,
        dist_metric="euclidean",
        use_metric_cuhk03=False,
        use_aqe=False,
        rerank=False,
        use_roc=False,
        logger=None,
    ):
        """Evaluate the ReID dataset

        - results: dict

        """
        assert self._is_eval, "ERR: not in eval mode"
        assert self._is_query_gallery, "ERR: dataset is not query/gallery mode"

        if metric_options is None:
            metric_options = dict(rank_list=[1, 5, 10, 20], max_rank=20)
        for rank in metric_options["rank_list"]:
            assert rank >= 1 and rank <= metric_options["max_rank"]
        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError("metric must be a list or a str")

        allowed_metrics = ["metric", "mAP", "CMC", "mINP"]
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f"metric {metric} is not supported.")

        # assert that results is list of tensors
        features = torch.stack(results)
        pids = self.get_pids()
        camids = self.get_camids()

        # separate query and gallery
        assert len(features) == len(self.data_infos)
        q_feat = features[: self._num_query]
        g_feat = features[self._num_query :]
        q_pids = pids[: self._num_query]
        g_pids = pids[self._num_query :]
        q_camids = camids[: self._num_query]
        g_camids = camids[self._num_query :]

        if normalize_features:
            # L2 normalization
            q_feat = F.normalize(q_feat, p=2, dim=-1)
            g_feat = F.normalize(g_feat, p=2, dim=-1)

        # results contains the keys:
        # 'CMC', 'mAP', 'mINP', 'metric', 'TPR@FPR', and 'Rank-#'
        # 'CMC' contains all ranks
        # rank-specific CMC are contained in 'Rank-#'
        # 'metric' = (mAP + cmc[0]) / 2 * 100

        results = evaluate(
            q_feat=q_feat,
            g_feat=g_feat,
            q_pids=q_pids,
            g_pids=g_pids,
            q_camids=q_camids,
            g_camids=g_camids,
            metric=dist_metric,
            ranks=metric_options["rank_list"],
            use_metric_cuhk03=use_metric_cuhk03,
            use_aqe=use_aqe,
            qe_times=1,
            qe_k=5,
            alpha=3.0,
            rerank=rerank,
            k1=20,
            k2=6,
            lambda_value=0.3,
            use_roc=use_roc,
        )

        eval_results = dict()

        # TODO: edit (finalize) returned results
        if "mAP" in metrics:
            eval_results["mAP"] = round(float(results["mAP"]), 4)
        if "CMC" in metrics:
            for rank in metric_options["rank_list"]:
                eval_results[f"Rank-{rank}"] = round(
                    float(results[f"Rank-{rank}"]),
                    4,
                )
        if "metric" in metrics:
            eval_results["metric"] = round(float(results["metric"]), 4)
        if "mINP" in metrics:
            eval_results["mINP"] = round(float(results["mINP"]), 4)

        return eval_results
