#!/usr/bin/env python3

import json
import warnings

import numpy as np

from .base import ImageDataset
from ..builder import DATASETS


@DATASETS.register_module()
class MOTImageDataset(ImageDataset):
    """MOT Image Dataset

    - support MOT16, MOT17, MOT20

    NOTE:
    - concat MOT16 with MOT17 will result in duplicates (same base images)
    """

    _mot_sequences = ()

    def __init__(
        self,
        train_seq,
        vis_ratio=0.7,
        **kwargs,
    ):
        # Do some checks!
        eval_mode = kwargs.pop("eval_mode", None)
        if eval_mode:
            warnings.warn("`eval_mode` cannot be enable for MOT datasets")
            eval_mode = False
        num_camids = kwargs.pop("num_camids", None)
        if num_camids is not None:
            warnings.warn("`num_camids` should not be set")
            num_camids = None

        if isinstance(train_seq, str):
            train_seq = train_seq
        for seq in train_seq:
            assert (
                seq in self._mot_sequences
            ), f"ERR: {seq} is not in {self._mot_sequences}"
        self.train_seq = train_seq

        assert 0 < vis_ratio <= 1
        self.vis_ratio = vis_ratio

        super().__init__(
            eval_mode=eval_mode,
            num_camids=num_camids,
            **kwargs,
        )

    def load_annotations(self):
        def _get_annotations(
            ann_file,
            data_prefix,
            train_seqs,
            vis_ratio,
        ):
            assert isinstance(ann_file, str)
            with open(ann_file, "r") as f:
                tmp_data = json.load(f)
            assert isinstance(tmp_data, list)
            data_infos = []
            for i, d in enumerate(tmp_data):
                pid = d["pid"]
                camid = d["camid"]
                img_path = d["img_path"]

                # mot specific:
                seq = d["seq"]
                vis_ratio = d["vis_ratio"]

                # filter out some samples
                if seq in train_seqs:
                    if vis_ratio >= self.vis_ratio:
                        info = dict(
                            img_prefix=data_prefix,
                            img_info=dict(
                                filename=img_path,
                                pid=pid,
                                camid=camid,
                                split="train",
                                debug_index=i,
                            ),
                            gt_label=np.array(pid, dtype=np.int64),
                        )
                        data_infos.append(info)
            del tmp_data
            return data_infos

        data_infos = _get_annotations(
            self.ann_file,
            self.data_prefix,
            self.train_seq,
            self.vis_ratio,
        )

        return data_infos


@DATASETS.register_module()
class MOT16ImageDataset(MOTImageDataset):
    """MOT16 Image Dataset"""

    _mot_sequences = ("02", "04", "05", "09", "10", "11", "13")


@DATASETS.register_module()
class MOT17ImageDataset(MOTImageDataset):
    """MOT17 Image Dataset"""

    _mot_sequences = ("02", "04", "05", "09", "10", "11", "13")


@DATASETS.register_module()
class MOT20ImageDataset(MOTImageDataset):
    """MOT17 Image Dataset"""

    _mot_sequences = ("01", "02", "03", "05")
