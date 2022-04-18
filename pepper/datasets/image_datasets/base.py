#!/usr/bin/env python3

from collections import defaultdict
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
        test_mode=False,
    ):
        super(ImageDataset, self).__init__(
            data_prefix=data_prefix,
            pipeline=pipeline,
            ann_file=ann_file,
            test_mode=test_mode,
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
            img_path = d["img_path"]
            info = dict(
                sampler_info=dict(
                    pid=pid,
                    camid=camid,
                ),
                img_prefix=self.data_prefix,
                img_info=dict(
                    filename=img_path,
                    camid=camid,
                    debug_index=i,  # FIXME: debugging
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

    def evaluate(self, results, metric="mAP", metric_options=None, logger=None):
        ...
