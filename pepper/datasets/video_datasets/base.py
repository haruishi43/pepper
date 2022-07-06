#!/usr/bin/env python3

import json
import warnings

import numpy as np
import torch

from ..builder import DATASETS
from ..base_dataset import BaseDataset


@DATASETS.register_module()
class VideoDataset(BaseDataset):
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
                        split=mode,
                        debug_index=i,
                    ),
                    gt_label=np.array(pid, dtype=np.int64),
                )
                data_infos.append(info)
            del tmp_data
            return data_infos

        if self._is_query_gallery:
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
        else:
            # training split
            data_infos = _get_annotations(self.ann_file, self.data_prefix)

        return data_infos

    def prepare_data(self, data):
        """Prepare data sample before handing it to pipeline

        Pipelines are designed to take list of dictionaries for sequential data.
        This means that we need a dictionary for each frame in the sequence.
        """
        assert isinstance(data, dict)
        img_prefix = data["img_prefix"]
        img_info = data["img_info"]
        gt_label = data["gt_label"]

        # make a list of dicts
        filenames = img_info["filenames"]
        results = []
        for i, fn in enumerate(filenames):
            info = dict(
                img_prefix=img_prefix,
                img_info=dict(
                    filename=fn,
                    pid=img_info["pid"],
                    camid=img_info["camid"],
                    frame_id=i,
                    is_video_data=True,
                    split=img_info["split"],
                    debug_index=img_info["debug_index"],
                ),
                gt_label=gt_label,
            )
            results.append(info)

        return self.pipeline(results)

    def evaluate(
        self,
        results,
        reduction="average",
        **kwargs,
    ):
        """For sequential data, we need a better way of obtaining features"""

        assert isinstance(results, list)

        sample = results[0]

        if sample.squeeze(0).dim() > 1:
            # multi-dim features cannot be evaluated directly,
            # we need to reduce the features to single dim
            warnings.warn(
                f"Multi-dim features detected (shape: {sample.shape})"
                "we need to reduce the features to single dim."
                f"Using {reduction} reduction method."
            )

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

        return super(VideoDataset, self).evaluate(
            results=results,
            **kwargs,
        )
