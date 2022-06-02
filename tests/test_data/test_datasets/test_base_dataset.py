#!/usr/bin/env python3

from mmcv import Config
from mmcv.utils import build_from_cfg

from pepper.datasets import BaseDataset, DATASETS, build_dataset
from pepper.core.evaluation.reid_evaluation import evaluate

"""Goals of this test suite

Test evaluation metric

"""


def test_mini_market1501():

    DO_TEST = False
    cfg = Config.fromfile("tests/configs/_base_/datasets/mini_market1501.py")

    print(cfg.pretty_text)

    print("creating training set")
    train_set = build_from_cfg(cfg.data.train, DATASETS)

    print("creating val set")
    val_set = build_from_cfg(cfg.data.val, DATASETS)
    assert len(val_set.get_query_infos()) == 32
    assert len(val_set.get_gallery_infos()) == 32
    assert len(val_set.data_infos) == 64, "query + gallery"
    assert len(val_set.get_gt_labels()) == len(val_set.data_infos)

    pids = val_set.get_pids()
    camids = val_set.get_camids()
    print(len(pids))
    print(pids)
    print(len(camids))
    print(camids)

    if DO_TEST:
        print("creating test set")
        test_set = build_from_cfg(cfg.data.test, DATASETS)
        assert len(test_set.get_query_infos()) == 32
        assert len(test_set.get_gallery_infos()) == 32


# def test_mini_mars():
#     cfg = Config.fromfile("tests/configs/mini_mars.py")

#     print(cfg.pretty_text)

#     dataset = build_from_cfg(cfg.data.train, DATASETS)
