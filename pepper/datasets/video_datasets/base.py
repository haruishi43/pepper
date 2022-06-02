#!/usr/bin/env python3

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

        NOTE: emphasis on the 's' in some keys
        """

        def _get_annotations(
            ann_file,
            data_prefix,
            mode="train",
        ):
            assert isinstance(ann_file, str)
            with open(ann_file, "r") as f:
                tmp_data = json.load(f)
            assert isinstance(tmp_data, list)
            data_infos = []
            for i, d in enumerate(tmp_data):
                pid = d["pid"]
                camid = d["camid"]
                img_paths = d["img_paths"]  # emphasis on 's'
                info = dict(
                    img_prefix=data_prefix,
                    img_info=dict(
                        filenames=sorted(img_paths),  # emphasis on 's'
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

            # dataloading needs a single list sos we concat it
            data_infos = query_infos + gallery_infos
        return data_infos

    def prepare_data(self, data):
        """Prepare data sample before handing it to pipelein

        Pipelines are designed to take list of dictionaries for sequential data.
        This means that we need a dictionary for each frame in the sequence.
        """
        img_prefix = data["img_prefix"]
        info = data["img_info"]
        gt_label = data["gt_label"]

        # make a list of dicts
        filenames = info["filenames"]
        results = []
        for i, fn in enumerate(filenames):
            info = dict(
                img_prefix=img_prefix,
                img_info=dict(
                    filename=fn,
                    pid=info["pid"],
                    camid=info["camid"],
                    frame_id=i,
                    is_video_data=True,
                    debug_eval=info["debug_eval"],
                    debug_index=info["debug_index"],
                ),
                gt_label=gt_label,
            )
            results.append(info)

        return self.pipeline(results)

    def evaluate(
        self,
        results,
        reduction="flatten",
        **kwargs,
    ):
        """For sequential data, we need a better way of obtaining features"""

        # prepare the results here if it haven't yet
        if len(results) != len(self.data_infos):
            # reduce the features to single dim
            new_results = []
            for r in results:
                if reduction == "flatten":
                    r = torch.flatten(r)
                elif reduction == "average":
                    # take the average (assume that features are 2 dim)
                    assert (
                        len(r.shape) == 2
                    ), f"ERR: {r.shape} is not a valid dim for features"
                    r = torch.mean(r, dim=0)
                else:
                    raise ValueError(
                        f"ERR: {reduction} is not valid reduction method"
                    )
                new_results.append(r)

            results = new_results

        super(VideoDataset, self).evaluate(
            results=results,
            **kwargs,
        )
