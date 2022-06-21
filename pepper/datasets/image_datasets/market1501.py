#!/usr/bin/env python3

from .base import ImageDataset
from ..builder import DATASETS


@DATASETS.register_module()
class Market1501Dataset(ImageDataset):
    def evaluate(self, results, **kwargs):
        return super().evaluate(
            results,
            use_metric_cuhk03=False,
            **kwargs,
        )
