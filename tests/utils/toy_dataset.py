from unittest.mock import MagicMock, patch

from pepper.datasets import BaseDataset


@patch.multiple(BaseDataset, __abstractmethods__=set())
def construct_toy_image_dataset(
    length: int,
    num_ids: int = 16,
    num_camids: int = 5,
):
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
                    debug_mode="train",
                    debug_index=i,
                ),
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
):
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
                    debug_mode="train",
                    debug_index=i,
                ),
            )
        )

    # need to create data_infos (list[dict('img_info')])
    BaseDataset.__getitem__ = MagicMock(side_effect=lambda idx: data_infos[idx])
    dataset = BaseDataset(
        data_prefix="", pipeline=[], ann_file=None, eval_mode=False
    )
    dataset.data_infos = data_infos
    return dataset
