#!/usr/bin/env python3

import copy
import os.path as osp
from abc import ABCMeta, abstractmethod
from os import PathLike

import numpy as np
from torch.utils.data import Dataset

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
        test_mode (bool): in train mode or test mode
    """

    CLASSES = None

    def __init__(
        self,
        data_prefix,
        pipeline,
        ann_file=None,
        test_mode=False,
    ):
        super(BaseDataset, self).__init__()
        self.data_prefix = expanduser(data_prefix)
        self.pipeline = Compose(pipeline)
        self.ann_file = expanduser(ann_file)
        self.test_mode = test_mode

        # data_infos is a List of dicts
        # the dicts should contain 'sampler_info' and 'img_info'
        self.data_infos = self.load_annotations()

    @abstractmethod
    def load_annotations(self):
        pass

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    def get_gt_labels(self):
        gt_labels = np.array([data["gt_label"] for data in self.data_infos])
        return gt_labels

    @abstractmethod
    def evaluate(self):
        pass
