#!/usr/bin/env python3

from unittest.mock import MagicMock, patch

from pepper.datasets import BaseDataset, build_sampler


@patch.multiple(BaseDataset, __abstractmethods__=set())
def construct_toy_dataset(length: int, num_ids: int = 16, num_camids: int = 5):
    assert length > num_ids
    data_infos = []
    for i in range(length):
        data_infos.append(
            dict(
                sampler_info=dict(
                    pid=i % num_ids,
                    camid=i % num_camids,
                ),
                img_prefix="data",
                img_info=dict(
                    file_name=f"{str(i)}.jpg",
                    camid=i % num_camids,
                    debug_index=i,
                ),
            )
        )

    # need to create data_infos (list[dict('sampler_info')])
    BaseDataset.__getitem__ = MagicMock(side_effect=lambda idx: data_infos[idx])
    dataset = BaseDataset(data_prefix="", pipeline=[], ann_file=None, test_mode=False)
    dataset.data_infos = data_infos
    return dataset


def get_matches(l1, l2):
    assert len(l1) == len(l2)
    tot = len(l1)
    matches = []
    for i in range(tot):
        i1 = l1[i]
        i2 = l2[i]
        matches.append(i1 == i2)
    return matches


def test_native_sampler():
    length = 1000
    num_ids = 124
    num_camids = 4

    # construct a toy dataset
    dataset = construct_toy_dataset(
        length,
        num_ids=num_ids,
        num_camids=num_camids,
    )

    # build sampler
    sampler = build_sampler(
        dict(
            type="NaiveIdentitySampler",
            dataset=dataset,
        ),
    )

    assert len(sampler) == length

    sample1 = list(sampler.__iter__())
    sample2 = list(sampler.__iter__())

    print(len(sample1))

    matches = get_matches(sample1, sample2)
    assert sum(matches) < len(sample1)


def test_native_dist_sampler():
    num_replicas = 2
    length = 1000
    num_ids = 124
    num_camids = 4

    # construct a toy dataset
    dataset = construct_toy_dataset(
        length=length,
        num_ids=num_ids,
        num_camids=num_camids,
    )

    # build samplers
    sampler1 = build_sampler(
        dict(
            type="NaiveIdentityDistributedSampler",
            dataset=dataset,
            num_replicas=num_replicas,
            rank=0,
        )
    )
    sampler2 = build_sampler(
        dict(
            type="NaiveIdentityDistributedSampler",
            dataset=dataset,
            num_replicas=num_replicas,
            rank=0,
        )
    )

    assert len(sampler1) == length // 2

    sample1 = list(sampler1.__iter__())
    sample2 = list(sampler2.__iter__())

    # FIXME: what are reasonable tests?


if __name__ == "__main__":

    dataset = construct_toy_dataset(33)

    for data in dataset:
        print(data)
