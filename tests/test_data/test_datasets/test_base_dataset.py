#!/usr/bin/env python3

from mmcv import Config
from mmcv.utils import build_from_cfg

from pepper.datasets import DATASETS


def test_mini_market1501():
    cfg = Config.fromfile("tests/configs/mini_market1501.py")

    print(cfg.pretty_text)

    dataset = build_from_cfg(cfg.data.train, DATASETS)
