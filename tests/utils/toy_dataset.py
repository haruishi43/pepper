#!/usr/bin/env python3

from unittest.mock import MagicMock, patch

import numpy as np

from pepper.datasets import BaseDataset


@patch.multiple(BaseDataset, __abstractmethods__=set())
def construct_toy_image_dataset(
    length: int,
    num_ids: int = 16,
    num_camids: int = 5,
    split: str = "train",
):
    """Construct toy dataset for ImageDataset

    `length` (int): the dataset size
    `num_ids` (int): the number of identities
    `num_camids` (int): the number of camera identities
    `split` (str): choose from train, query, gallery
    """
    assert length > num_ids
    assert length < 100_000, "exceeds the total number of samples"
    data_infos = []
    for i in range(length):
        pid = i % num_ids
        camid = i % num_camids
        data_infos.append(
            dict(
                img_prefix="data",
                img_info=dict(
                    filename=f"{str(i).zfill(5)}.jpg",
                    pid=pid,
                    camid=camid,
                    split=split,
                    debug_index=i,
                ),
                gt_label=np.array(pid),
            )
        )

    # need to create data_infos (list[dict('img_info')])
    BaseDataset.__getitem__ = MagicMock(side_effect=lambda idx: data_infos[idx])
    dataset = BaseDataset(
        data_prefix="", pipeline=[], ann_file=None, eval_mode=False
    )
    dataset.data_infos = data_infos
    return dataset


@patch.multiple(BaseDataset, __abstractmethods__=set())
def construct_toy_video_dataset(
    length: int,
    num_ids: int = 16,
    num_camids: int = 5,
    num_frames: int = 8,
    split: str = "train",
):
    """Construct toy dataset for VideoDataset

    `length` (int): the dataset size
    `num_ids` (int): the number of identities
    `num_camids` (int): the number of camera identities
    `num_frames` (int): the number of frames per each sample
    """
    assert length > num_ids
    assert length < 100_000, "exceeds the total number of samples"
    data_infos = []
    for i in range(length):

        pid = i % num_ids
        camid = i % num_camids

        filenames = []
        for j in range(num_frames):
            filenames.append(f"{str(i).zfill(5)}_{str(j).zfill(5)}.jpg")

        data_infos.append(
            dict(
                img_prefix="data",
                img_info=dict(
                    filenames=filenames,
                    pid=pid,
                    camid=camid,
                    split=split,
                    debug_index=i,
                ),
                gt_label=np.array(),
            )
        )

    def _prepare_data(di):
        img_prefix = di["img_prefix"]
        info = di["img_info"]
        gt_label = di["gt_label"]

        filenames = info["filenames"]
        results = []
        for i, fn in enumerate(filenames):
            info = dict(
                img_prefix=img_prefix,
                img_info=dict(
                    filename=fn,
                    pid=info["pid"],
                    camid=info["camid"],
                    frame_id=i,
                    is_video_data=True,
                    split=info["split"],
                    debug_index=info["debug_index"],
                ),
                gt_label=gt_label,
            )
        return results

    # need to create data_infos (list[dict('img_info')])
    BaseDataset.__getitem__ = MagicMock(
        side_effect=lambda idx: _prepare_data(data_infos[idx])
    )
    dataset = BaseDataset(
        data_prefix="", pipeline=[], ann_file=None, eval_mode=False
    )
    dataset.data_infos = data_infos
    return dataset
