#!/usr/bin/env python3

"""Goals of this test suite

Test evaluation metric

"""

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from pepper.datasets import BaseDataset
from pepper.core.evaluation.reid_evaluation import evaluate


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

    # need to convert array to tensor
    query_feats = torch.from_numpy(query_feats.astype(np.float32)) / n
    gallery_feats = torch.from_numpy(gallery_feats.astype(np.float32)) / n
    # query_feats = F.normalize(query_feats, p=2, dim=0)
    # gallery_feats = F.normalize(gallery_feats, p=2, dim=0)

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

    return (
        query_pids,
        query_camids,
        query_feats,
        gallery_pids,
        gallery_camids,
        gallery_feats,
    )


if __name__ == "__main__":

    # args
    debug = False

    # extreme
    num_ids = 1500  # 4
    feature_length = 1024  # 2
    num_instance_gallery = 6  # 2
    factor = 10.0

    # easy
    num_ids = 4
    feature_length = 2
    num_instance_gallery = 2
    factor = 0.1

    # medium
    num_ids = 20
    feature_length = 8
    num_instance_gallery = 4
    factor = 0.5

    # create query and gallery predictions and
    (
        query_pids,
        query_camids,
        query_feats,
        gallery_pids,
        gallery_camids,
        gallery_feats,
    ) = create_toy_reid_dataset(
        n=num_ids,
        nfeat=feature_length,
        ninst=num_instance_gallery,
        factor=factor,
        debug=debug,
    )

    # evaluate predictions
    # results = evaluate(
    #     q_feat=query_feats,
    #     g_feat=gallery_feats,
    #     q_pids=query_pids,
    #     g_pids=gallery_pids,
    #     q_camids=query_camids,
    #     g_camids=gallery_camids,
    #     metric="euclidean",
    #     ranks=[1],
    # )
    # print(results["mAP"])

    # when pids=5000, feat=32, inst=12, we can see improvements

    results = evaluate(
        q_feat=query_feats,
        g_feat=gallery_feats,
        q_pids=query_pids,
        g_pids=gallery_pids,
        q_camids=query_camids,
        g_camids=gallery_camids,
        metric="euclidean",
        ranks=[1],
        max_rank=10,
        # use_aqe=True,
        # rerank=True,
        # use_roc=True,
    )

    print(results["mAP"])
    print(results["metric"])
    print(results["CMC"])
    print(len(results["CMC"]))
