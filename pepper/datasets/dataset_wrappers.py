#!/usr/bin/env python3

"""
Only for training!
"""

from copy import deepcopy

import numpy as np
from torch.utils.data import Dataset

from .builder import DATASETS


@DATASETS.register_module()
class ConcatTrainDataset(Dataset):
    """A wrapper of concatenated dataset."""

    NUM_PIDS = None
    NUM_CAMIDS = None

    pipeline = None

    def __init__(self, datasets):
        super(ConcatTrainDataset, self).__init__()

        assert (
            len(datasets) > 1
        ), f"ConcatDataset needs more than 1 dataset {len(datasets)}"

        # we can add pipeline to data_infos and pop them before we run the pipeline
        # self.pipeline = datasets[0].pipeline
        self.data_infos = self.refine_data_infos(datasets)

        # add globals
        self.NUM_PIDS = len(np.unique(self.get_pids()))
        self.NUM_CAMIDS = len(np.unique(self.get_camids()))

    def refine_data_infos(self, datasets):
        data_infos = []
        index = 0
        cum_pids = 0
        cum_camids = 0
        for dataset in datasets:
            di = deepcopy(dataset.data_infos)
            for info in di:
                # FIXME: ugly...
                img_info = deepcopy(info["img_info"])
                img_info["pid"] = cum_pids + img_info["pid"]

                # FIXME: if we're using query/gallery of the same dataset,
                # we're going to treat the camids as completely new.
                # this should affect the training since pids are also completely
                # different for many datasets
                img_info["camid"] = cum_camids + img_info["camid"]
                img_info["debug_index"] = index + img_info["debug_index"]

                new_info = dict(
                    img_prefix=info["img_prefix"],
                    img_info=img_info,
                    gt_label=np.array(img_info["pid"], dtype=np.int64),
                    pipeline=dataset.pipeline,
                )
                data_infos.append(new_info)
                index += 1

            cum_pids += len(np.unique(dataset.get_pids()))  # len of numpy array
            cum_camids += len(
                np.unique(dataset.get_camids())
            )  # len of numpy array

        return data_infos

    def prepare_data(self, data):
        # NOTE: might need to prepare the sample before handing it to pipeline
        # different pipelines for each dataset
        if self.pipeline is None:
            pipeline = data.pop("pipeline")
            return pipeline(data)
        else:
            return self.pipeline(data)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):

        # NOTE: some functions in the pipeline may be inplace
        data = deepcopy(self.data_infos[idx])

        return self.prepare_data(data)

    def get_gt_labels(self):
        gt_labels = np.array([data["gt_label"] for data in self.data_infos])
        return gt_labels

    def get_pids(self):
        infos = deepcopy(self.data_infos)
        return np.asarray([info["img_info"]["pid"] for info in infos])

    def get_camids(self):
        infos = deepcopy(self.data_infos)
        return np.asarray([info["img_info"]["camid"] for info in infos])
