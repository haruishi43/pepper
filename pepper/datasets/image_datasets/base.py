#!/usr/bin/env python3

import json

import numpy as np
import torch

from ..builder import DATASETS
from ..base_dataset import BaseDataset


@DATASETS.register_module()
class ImageDataset(BaseDataset):
    def __init__(
        self,
        data_prefix,
        pipeline,
        ann_file=None,
        eval_mode=False,
    ):
        super(ImageDataset, self).__init__(
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

        def _get_annotations(ann_file, data_prefix, mode="train",):
            assert isinstance(ann_file, str)
            with open(ann_file, "r") as f:
                tmp_data = json.load(f)
            assert isinstance(tmp_data, list)
            data_infos = []
            for i, d in enumerate(tmp_data):
                pid = d["pid"]
                camid = d["camid"]
                img_path = d["img_path"]
                info = dict(
                    img_prefix=data_prefix,
                    img_info=dict(
                        filename=img_path,
                        pid=pid,
                        camid=camid,
                        debug_eval=mode,
                        debug_index=i,
                    ),
                    gt_label=np.array(pid, dtype=np.int64),
                )
                data_infos.append(info)
            del tmp_data
            return data_infos

        if not self._is_eval:
            data_infos = _get_annotations(self.ann_file, self.data_prefix)
        else:
            query_infos = _get_annotations(
                self.ann_file["query"],
                self.data_prefix["query"],
                mode="query",
            )
            self._num_query = len(query_infos)
            gallery_infos = _get_annotations(
                self.ann_file["gallery"],
                self.data_prefix["gallery"],
                mode="gallery",
            )
            self._num_gallery = len(gallery_infos)
            data_infos = query_infos + gallery_infos
        return data_infos

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

        # distance

        #
