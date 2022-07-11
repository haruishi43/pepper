#!/usr/bin/env python3

import json

import numpy as np

from ..builder import DATASETS
from ..base_dataset import BaseDataset


@DATASETS.register_module()
class ImageDataset(BaseDataset):
    def load_annotations(self):
        """Load annotations from ImageNet style annotation file.
        Returns:
            list[dict]: Annotation information from ReID api.
        """

        # TODO: mode `_get_annotations` to another file so it can be extensible

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
                img_path = d["img_path"]
                info = dict(
                    img_prefix=data_prefix,
                    img_info=dict(
                        filename=img_path,
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
            # query/gallery pairs
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

            # dataloading needs a single list so we concat it
            data_infos = query_infos + gallery_infos
        else:
            # training split
            data_infos = _get_annotations(self.ann_file, self.data_prefix)

        return data_infos
