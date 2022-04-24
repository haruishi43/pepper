#!/usr/bin/env python3

from unittest.mock import MagicMock, patch

import pytest

import numpy as np
import torch

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
    ninst: int = 10,
    factor: float = 0.1,
    seed: int = 0,
    debug: bool = False,  # print out
) -> np.ndarray:
    _rng = np.random.default_rng(seed)

    # query
    query_pids = np.array(list(range(n)), dtype=np.int64)  # (n)
    query_camids = np.array([0] * n, dtype=np.int64)  # (n)
    query_feats = []
    for i in range(n):
        query_feats.append(_rng.normal(i, 1, nfeat))
    query_feats = np.stack(query_feats, axis=0)  # (n, nfeat)

    # gallery
    num_instances = ninst
    gallery_pids = []
    gallery_camids = []
    gallery_feats = []
    for i in range(n):
        for j in range(num_instances):
            gallery_pids.append(i)
            gallery_camids.append(j + 1)

            # make the gallery features close to the query
            gallery_feats.append(
                query_feats[i] - factor * _rng.normal(0, 1, nfeat)
            )
    gallery_pids = np.asarray(gallery_pids)  # (num_instances * n)
    gallery_camids = np.asarray(gallery_camids)  # (num_instances * n)
    gallery_feats = np.stack(
        gallery_feats, axis=0
    )  # (num_instances * n, nfeat)

    # debug print outs
    if debug:
        print("query pids", query_pids.shape)
        print("query camids", query_camids.shape)
        print("query feats", query_feats.shape)
        print("gallery pids", gallery_pids.shape)
        print("gallery camids", gallery_camids.shape)
        print("gallery feats", gallery_feats.shape)

        i_inst = 0
        for i in range(n):
            print("-" * 16)
            print("pids:", query_pids[i])
            print("camids:", query_camids[i])

            for j in range(num_instances):
                print("gallery camids:", gallery_camids[i_inst])
                print("query_feat:", query_feats[i])
                print(f"gallery_feat #{i_inst}:", gallery_feats[i_inst])
                i_inst += 1

    return query_pids, query_camids, query_feats, gallery_pids, gallery_camids, gallery_feats


if __name__ == "__main__":

    # create query and gallery predictions and
    (
        query_pids,
        query_camids,
        query_feats,
        gallery_pids,
        gallery_camids,
        gallery_feats,
    ) = create_toy_reid_dataset(
        n=100,
        nfeat=2,
        ninst=2,
        factor=0.3,
    )

    # evaluate predictions
    query_feats = torch.from_numpy(query_feats)
    gallery_feats = torch.from_numpy(gallery_feats)

    results = evaluate(
        q_feat=query_feats,
        g_feat=gallery_feats,
        q_pids=query_pids,
        g_pids=gallery_pids,
        q_camids=query_camids,
        g_camids=gallery_camids,
        metric="euclidean",
        ranks=[1],
    )

    print(results["mAP"])
