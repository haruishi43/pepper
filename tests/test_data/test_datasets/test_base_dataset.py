#!/usr/bin/env python3

import numpy as np
from unittest.mock import MagicMock, patch

import pytest

from mmcv import Config
from mmcv.utils import build_from_cfg

from pepper.datasets import BaseDataset, DATASETS, build_dataset

"""Goals of this test suite

Test evaluation metric

"""


def test_mini_market1501():
    cfg = Config.fromfile("tests/configs/_base_/datasets/mini_market1501.py")

    print(cfg.pretty_text)

    print("creating training set")
    train_set = build_from_cfg(cfg.data.train, DATASETS)

    print("creating val set")
    val_set = build_from_cfg(cfg.data.val, DATASETS)
    assert len(val_set.get_query_infos()) == 32
    assert len(val_set.get_gallery_infos()) == 32

    print("creating test set")
    test_set = build_from_cfg(cfg.data.test, DATASETS)
    assert len(test_set.get_query_infos()) == 32
    assert len(test_set.get_gallery_infos()) == 32


# def test_mini_mars():
#     cfg = Config.fromfile("tests/configs/mini_mars.py")

#     print(cfg.pretty_text)

#     dataset = build_from_cfg(cfg.data.train, DATASETS)


# evaluation

@patch.multiple(BaseDataset, __abstractmethods__=set())
def construct_toy_dataset(length: int, num_ids: int = 16, num_camids: int = 5):
    assert length > num_ids
    data_infos = []
    for i in range(length):
        data_infos.append(
            dict(
                img_prefix="data",
                img_info=dict(
                    file_name=f"{str(i)}.jpg",
                    pid=i % num_ids,
                    camid=i % num_camids,
                    debug_index=i,
                ),
            )
        )

    BaseDataset.__getitem__ = MagicMock(side_effect=lambda idx: data_infos[idx])
    dataset = BaseDataset(
        data_prefix="", pipeline=[], ann_file=None, eval_mode=False
    )
    dataset.data_infos = data_infos
    return dataset


def create_toy_reid_dataset(
    n: int = 4,
    nfeat: int = 2,
    seed: int = 0,
) -> np.ndarray:
    _rng = np.random.default_rng(seed)

    # query
    query_labels = np.array(list(range(n)), dtype=np.int64)  # (n)
    query_feats = []
    for i in range(n):
        query_feats.append(_rng.normal(i, 1, nfeat))
    query_feats = np.stack(query_feats, axis=0)  # (n, nfeat)

    # gallery
    num_instances = 2
    gallery_labels = []
    gallery_feats = []
    for i in range(n):
        for _ in range(num_instances):
            gallery_labels.append(i)

            # make the gallery features close to the query
            gallery_feats.append(
                query_feats[i] - 0.1 * _rng.normal(0, 1, nfeat)
            )
    gallery_labels = np.stack(gallery_labels, axis=0)  # (num_instances * n)
    gallery_feats = np.stack(
        gallery_feats, axis=0
    )  # (num_instances * n, nfeat)

    # debug print outs
    print("query labels", query_labels.shape)
    print("query feats", query_feats.shape)
    print("gallery labels", gallery_labels.shape)
    print("gallery feats", gallery_feats.shape)

    i_inst = 0
    for i in range(n):
        print("-" * 16)
        print("label:", query_labels[i])

        print("query_feat:", query_feats[i])
        print(f"gallery_feat #{i_inst}:", gallery_feats[i_inst])
        i_inst += 1
        print(f"gallery_feat #{i_inst}:", gallery_feats[i_inst])
        i_inst += 1

    return query_labels, query_feats, gallery_labels, gallery_feats


if __name__ == "__main__":

    # create query and gallery predictions and
    (
        query_labels,
        query_feats,
        gallery_labels,
        gallery_feats,
    ) = create_toy_reid_dataset(
        n=4,
        nfeat=2,
    )

    # evaluate predictions
