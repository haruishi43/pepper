#!/usr/bin/env python3

import json
import warnings

import numpy as np

from .base import VideoDataset
from ..builder import DATASETS


@DATASETS.register_module()
class MOTVideoDataset(VideoDataset):
    """MOT Video Dataset

    - support MOT16, MOT17, MOT20

    NOTE:
    - concat MOT16 with MOT17 will result in duplicates (same base images)
    """

    _mot_sequences = ()

    def __init__(
        self,
        train_seq,
        vis_ratio=0.7,
        vis_frame_ratio=0.6,  # ratio of frames that satisfies vis_ratio
        min_seq_len=8,
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

        if train_seq is None:
            seq = self._mot_sequences
        if isinstance(train_seq, str):
            train_seq = train_seq
        for seq in train_seq:
            assert (
                seq in self._mot_sequences
            ), f"ERR: {seq} is not in {self._mot_sequences}"
        self.train_seq = train_seq

        assert 0 < vis_ratio <= 1
        self.vis_ratio = vis_ratio
        assert 0 < vis_ratio <= 1
        self.vis_frame_ratio = vis_frame_ratio

        self.min_seq_len = min_seq_len

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
            vis_frame_ratio,
            min_seq_len,
        ):
            assert isinstance(ann_file, str)
            with open(ann_file, "r") as f:
                tmp_data = json.load(f)
            assert isinstance(tmp_data, list)
            data_infos = []
            pid_container = set()
            for i, d in enumerate(tmp_data):
                pid = d["pid"]
                camid = d["camid"]  # None
                img_paths = d["img_paths"]

                # TODO: randomize camid for sampler?
                camid = 0

                # mot specific:
                seq = d["seq"]
                vis_ratios = d["vis_ratios"]

                # filter out some samples
                if seq in train_seqs:
                    if len(img_paths) >= min_seq_len:
                        ratio = np.sum(np.array(vis_ratios) >= vis_ratio) / len(
                            vis_ratios
                        )
                        if ratio >= vis_frame_ratio:
                            info = dict(
                                img_prefix=data_prefix,
                                img_info=dict(
                                    filenames=sorted(img_paths),
                                    vis_ratios=vis_ratios,
                                    pid=pid,
                                    camid=camid,
                                    split="train",
                                    debug_index=i,
                                ),
                                gt_label=np.array(pid, dtype=np.int64),
                            )
                            data_infos.append(info)
                            pid_container.add(pid)

            # relabel
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            for d in data_infos:
                pid = pid2label[d["img_info"]["pid"]]
                d["img_info"]["pid"] = pid
                d["gt_label"] = np.array(pid, dtype=np.int64)

            del tmp_data
            del pid2label
            return data_infos

        data_infos = _get_annotations(
            self.ann_file,
            self.data_prefix,
            self.train_seq,
            vis_ratio=self.vis_ratio,
            vis_frame_ratio=self.vis_frame_ratio,
            min_seq_len=self.min_seq_len,
        )

        return data_infos


@DATASETS.register_module()
class MOT16VideoDataset(MOTVideoDataset):
    """MOT16 Video Dataset"""

    _mot_sequences = ("02", "04", "05", "09", "10", "11", "13")


@DATASETS.register_module()
class MOT17VideoDataset(MOTVideoDataset):
    """MOT17 Video Dataset"""

    _mot_sequences = ("02", "04", "05", "09", "10", "11", "13")


@DATASETS.register_module()
class MOT20VideoDataset(MOTVideoDataset):
    """MOT17 Video Dataset"""

    _mot_sequences = ("01", "02", "03", "05")
