#!/usr/bin/env python3

from collections import defaultdict
import copy
import json

import numpy as np
import torch

from ..builder import DATASETS
from ..base_dataset import BaseDataset


@DATASETS.register_module()
class VideoDataset(BaseDataset):
    def __init__(
        self,
        data_prefix,
        pipeline,
        ann_file=None,
        eval_mode=False,
    ):
        super(VideoDataset, self).__init__(
            data_prefix=data_prefix,
            pipeline=pipeline,
            ann_file=ann_file,
            eval_mode=eval_mode,
        )

    def load_annotations(self):
        """Load annotations from ImageNet style annotation file.
        Returns:
            list[dict]: Annotation information from ReID api.
        """
        assert isinstance(self.ann_file, str)

        with open(self.ann_file, "r") as f:
            tmp_data = json.load(f)

        assert isinstance(tmp_data, list)
        data_infos = []
        for i, d in enumerate(tmp_data):
            pid = d["pid"]
            camid = d["camid"]
            img_paths = d["img_paths"]
            info = dict(
                img_prefix=self.data_prefix,
                img_info=dict(
                    filenames=sorted(img_paths),
                    pid=pid,
                    camid=camid,
                    debug_index=i,
                ),
            )
            info["gt_label"] = np.array(pid, dtype=np.int64)
            data_infos.append(info)

        del tmp_data

        if not self.test_mode:
            # relabel
            self._parse_ann_info(data_infos)
        return data_infos

    def _parse_ann_info(self, data_infos):
        """Parse person id annotations."""

        index_tmp_dic = defaultdict(list)
        self.index_dic = dict()
        for idx, info in enumerate(data_infos):
            pid = info["gt_label"]
            index_tmp_dic[int(pid)].append(idx)
        for pid, idxs in index_tmp_dic.items():
            self.index_dic[pid] = np.asarray(idxs, dtype=np.int64)

        self.pids = np.asarray(list(self.index_dic.keys()), dtype=np.int64)

    def prepare_data(self, data):
        """Prepare results for image (e.g. the annotation information, ...)."""
        info = data["img_info"]

        # make a list of dicts
        filenames = info["filenames"]
        results = []
        for i, fn in enumerate(filenames):
            info = dict(
                img_prefix=self.data_prefix,
                img_info=dict(
                    filename=fn,
                    pid=info["pid"],
                    camid=info["camid"],
                    frame_id=i,
                    is_video_data=True,
                ),
            )
            info["gt_label"] = copy.deepcopy(data_info["gt_label"])
            results.append(info)

        return self.pipeline(results)

    def evaluate(self, results, metric="mAP", metric_options=None, logger=None):
        """Evaluate the ReID dataset.
        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `mAP`.
            metric_options: (dict, optional): Options for calculating metrics.
                Allowed keys are 'rank_list' and 'max_rank'. Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = dict(rank_list=[1, 5, 10, 20], max_rank=20)
        for rank in metric_options["rank_list"]:
            assert rank >= 1 and rank <= metric_options["max_rank"]
        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError("metric must be a list or a str.")
        allowed_metrics = ["mAP", "CMC"]
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f"metric {metric} is not supported.")

        # distance
        results = [result.data.cpu() for result in results]
        features = torch.stack(results)

        n, c = features.size()
        mat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
        distmat = mat + mat.t()
        distmat.addmm_(features, features.t(), beta=1, alpha=-2)
        distmat = distmat.numpy()

        pids = self.get_gt_labels()
        indices = np.argsort(distmat, axis=1)
        matches = (pids[indices] == pids[:, np.newaxis]).astype(np.int32)

        all_cmc = []
        all_AP = []
        num_valid_q = 0.0
        for q_idx in range(n):
            # remove self
            raw_cmc = matches[q_idx][1:]
            if not np.any(raw_cmc):
                # this condition is true when query identity
                # does not appear in gallery
                continue

            cmc = raw_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[: metric_options["max_rank"]])
            num_valid_q += 1.0

            # compute average precision
            # reference:
            # https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = raw_cmc.sum()
            tmp_cmc = raw_cmc.cumsum()
            tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert (
            num_valid_q > 0
        ), "Error: all query identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        eval_results = dict()
        if "mAP" in metrics:
            eval_results["mAP"] = np.around(mAP, decimals=3)
        if "CMC" in metrics:
            for rank in metric_options["rank_list"]:
                eval_results[f"R{rank}"] = np.around(
                    all_cmc[rank - 1], decimals=3
                )

        return eval_results
