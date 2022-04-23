#!/usr/bin/env python3

from collections import Counter
import math
from unittest.mock import MagicMock, patch

import pytest

from pepper.datasets import BaseDataset, build_sampler


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


def get_matches(l1, l2):
    assert len(l1) == len(l2)
    tot = len(l1)
    matches = []
    for i in range(tot):
        i1 = l1[i]
        i2 = l2[i]
        matches.append(i1 == i2)
    return matches


# @pytest.mark.skip()
@pytest.mark.parametrize("length", [1000, 10_000])
@pytest.mark.parametrize("num_ids", [124, 17])
@pytest.mark.parametrize("num_camids", [4])
def test_native_sampler(length, num_ids, num_camids):
    batch_size = 32
    num_instances = 4

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
            batch_size=batch_size,
            num_instances=num_instances,
        ),
    )

    assert len(sampler) == length

    sample1 = list(sampler.__iter__())
    sample2 = list(sampler.__iter__())

    assert len(sample1) == len(sample2)
    assert len(sample1) == batch_size * math.ceil(
        num_ids / (batch_size // num_instances)
    )

    matches = get_matches(sample1, sample2)
    assert sum(matches) < len(sample1)

    data1 = [dataset[i] for i in sample1]
    pids1 = [d["img_info"]["pid"] for d in data1]
    pc = Counter(pids1)
    assert len(pc) == num_ids

    if 2 * num_instances < length / num_ids:
        ic = Counter(sample1)
        assert len(ic) == len(sample1)

    for i in range(10):
        sample = list(sampler.__iter__())
        inter = set(sample1).intersection(set(sample))
        diff = set(sample1).difference(set(sample))
        # print(i, len(inter), len(diff))
        assert len(inter) < len(sample)
        assert len(diff) > 0


# @pytest.mark.skip()
@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("num_instances", [4, 8])
@pytest.mark.parametrize("length", [1000, 10_000])
@pytest.mark.parametrize("num_ids", [17, 124])
@pytest.mark.parametrize("num_camids", [4, 6])
def test_native_dist_sampler(
    batch_size,
    num_instances,
    length,
    num_ids,
    num_camids,
):
    num_replicas = 2  # FIXME: add tests for larger world size

    # construct a toy dataset
    dataset = construct_toy_dataset(
        length=length,
        num_ids=num_ids,
        num_camids=num_camids,
    )

    # build samplers
    shuffle = True
    round_up = True
    sampler1 = build_sampler(
        dict(
            type="NaiveIdentityDistributedSampler",
            dataset=dataset,
            num_replicas=num_replicas,
            rank=0,
            batch_size=batch_size,
            num_instances=num_instances,
            shuffle=shuffle,
            round_up=round_up,
        )
    )
    sampler2 = build_sampler(
        dict(
            type="NaiveIdentityDistributedSampler",
            dataset=dataset,
            num_replicas=num_replicas,
            rank=1,
            batch_size=batch_size,
            num_instances=num_instances,
            shuffle=shuffle,
            round_up=round_up,
        )
    )

    assert len(sampler1) == length // 2

    for epoch in range(5):
        sampler1.set_epoch(epoch)
        sampler2.set_epoch(epoch)

        sample1 = list(sampler1.__iter__())
        sample2 = list(sampler2.__iter__())

        data1 = [dataset[i] for i in sample1]
        data2 = [dataset[i] for i in sample2]

        pids1 = [d["img_info"]["pid"] for d in data1]
        pids2 = [d["img_info"]["pid"] for d in data2]

        # pc1 = Counter(pids1)
        # pc2 = Counter(pids2)
        # print()
        # print(pc1.keys())
        # print(pc2.keys())

        num_same_ids = len(set(pids1).intersection(set(pids2)))

        ids_per_batch = batch_size // num_instances
        iterations = math.ceil(num_ids / ids_per_batch)
        assert (
            num_same_ids == iterations * ids_per_batch - num_ids
        ), "number of repeated ids is unexpected"


# @pytest.mark.skip()
@pytest.mark.parametrize("length", [1000, 10_000])
@pytest.mark.parametrize("num_ids", [124, 17])
@pytest.mark.parametrize("num_camids", [6])
def test_balanced_sampler(length, num_ids, num_camids):
    batch_size = 32
    num_instances = 4

    # construct a toy dataset
    dataset = construct_toy_dataset(
        length,
        num_ids=num_ids,
        num_camids=num_camids,
    )

    # build sampler
    sampler = build_sampler(
        dict(
            type="BalancedIdentitySampler",
            dataset=dataset,
            batch_size=batch_size,
            num_instances=num_instances,
        ),
    )

    assert len(sampler) == length

    sample1 = list(sampler.__iter__())
    sample2 = list(sampler.__iter__())

    assert len(sample1) == len(sample2)
    assert len(sample1) == batch_size * math.ceil(
        num_ids / (batch_size // num_instances)
    )

    matches = get_matches(sample1, sample2)
    assert sum(matches) < len(sample1)

    data1 = [dataset[i] for i in sample1]
    pids1 = [d["img_info"]["pid"] for d in data1]
    camids1 = [d["img_info"]["camid"] for d in data1]
    pc = Counter(pids1)
    cc = Counter(camids1)
    # print()
    # print(pc)
    # print(cc)
    assert len(pc) == num_ids, "should use all pids"
    assert len(cc) == num_camids, "should use all camids"

    if 2 * num_instances < math.floor(length / num_ids):
        # if there are many unique instances
        # FIXME: there are times where index repeats, need to debug those cases
        ic = Counter(sample1)
        assert len(ic) == len(sample1), "should not repeat index"

    for i in range(10):
        sample = list(sampler.__iter__())
        inter = set(sample1).intersection(set(sample))
        diff = set(sample1).difference(set(sample))
        # print(i, len(inter), len(diff))
        # inter should be low
        # diff should be high
        assert len(inter) < len(sample)
        assert len(diff) > 0


# FIXME: add real tests
# @pytest.mark.skip()
@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("num_instances", [4, 8])
@pytest.mark.parametrize("length", [1000, 10_000])
@pytest.mark.parametrize("num_ids", [17, 124])
@pytest.mark.parametrize("num_camids", [4, 6])
def test_balanced_dist_sampler(
    batch_size,
    num_instances,
    length,
    num_ids,
    num_camids,
):
    num_replicas = 2

    # construct a toy dataset
    dataset = construct_toy_dataset(
        length=length,
        num_ids=num_ids,
        num_camids=num_camids,
    )

    # build samplers
    sampler1 = build_sampler(
        dict(
            type="BalancedIdentityDistributedSampler",
            dataset=dataset,
            num_replicas=num_replicas,
            rank=0,
            batch_size=batch_size,
            num_instances=num_instances,
            # shuffle=False,
            # round_up=False,
        )
    )
    sampler2 = build_sampler(
        dict(
            type="BalancedIdentityDistributedSampler",
            dataset=dataset,
            num_replicas=num_replicas,
            rank=1,
            batch_size=batch_size,
            num_instances=num_instances,
            # shuffle=False,
            # round_up=False,
        )
    )

    assert len(sampler1) == length // 2

    for epoch in range(5):
        sampler1.set_epoch(epoch)
        sampler2.set_epoch(epoch)

        sample1 = list(sampler1.__iter__())
        sample2 = list(sampler2.__iter__())

        data1 = [dataset[i] for i in sample1]
        data2 = [dataset[i] for i in sample2]

        pids1 = [d["img_info"]["pid"] for d in data1]
        pids2 = [d["img_info"]["pid"] for d in data2]

        # pc1 = Counter(pids1)
        # pc2 = Counter(pids2)
        # print()
        # print(pc1.keys())
        # print(pc2.keys())

        num_same_ids = len(set(pids1).intersection(set(pids2)))

        ids_per_batch = batch_size // num_instances
        iterations = math.ceil(num_ids / ids_per_batch)
        assert (
            num_same_ids == iterations * ids_per_batch - num_ids
        ), "number of repeated ids is unexpected"


if __name__ == "__main__":

    dataset = construct_toy_dataset(33)

    for data in dataset:
        print(data)
