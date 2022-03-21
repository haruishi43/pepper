#!/usr/bin/env python3

import random

import numpy as np

from ..builder import PIPELINES

__all__ = ["VideoSampler"]


@PIPELINES.register_module()
class VideoSampler:
    def __init__(self, method: str):
        """Video Sampler
        params:
        - method: string
        Available methods:
        - all
        - evenly
        - random
        - random_crop
        """
        self._sample_func = getattr(self, method, None)
        assert (
            self._sample_func is not None
        ), f"ERR: {method} is not a valid sampling method"

    def __call__(self, num_imgs: int, seq_len: int, **kwargs) -> np.ndarray:
        return self._sample_func(num_imgs, seq_len, **kwargs)

    def all(self, num_imgs: int, seq_len: int) -> np.ndarray:
        r"""Just returns all images"""
        return np.arange(num_imgs)

    def evenly(self, num_imgs: int, seq_len: int) -> np.ndarray:
        """Evenly Sample
        Evenly samples seq_len images from a tracklet
        """
        if num_imgs >= seq_len:
            num_imgs -= num_imgs % seq_len
            indices = np.arange(0, num_imgs, num_imgs / seq_len)
        else:
            # if num_imgs is smaller than seq_len, simply replicate the last image
            # until the seq_len requirement is satisfied
            indices = np.arange(0, num_imgs)
            num_pads = seq_len - num_imgs
            indices = np.concatenate(
                [indices, np.ones(num_pads).astype(np.int32) * (num_imgs - 1)]
            )
        assert len(indices) == seq_len
        return indices

    def random(
        self, num_imgs: int, seq_len: int, sort: bool = True
    ) -> np.ndarray:
        """Random Sample
        Randomly samples seq_len images from a tracklet of length num_imgs,
        if num_imgs is smaller than seq_len, then replicates images
        """
        indices = np.arange(num_imgs)
        replace = False if num_imgs >= seq_len else True
        indices = np.random.choice(
            indices,
            size=seq_len,
            replace=replace,
        )
        # sort indices to keep temporal order (comment it to be order-agnostic)
        if sort:
            indices = np.sort(indices)
        return indices

    def random_crop(self, num_imgs: int, seq_len: int) -> np.ndarray:
        """Random Crop Sample
        Crop consecutive images from a tracklet
        """
        if num_imgs >= seq_len:
            initial_frame = random.randint(0, num_imgs - seq_len)
            indices = np.arange(initial_frame, initial_frame + seq_len)
        else:
            # if num_imgs is smaller than seq_len, simply replicate the last image
            # until the seq_len requirement is satisfied
            indices = np.arange(0, num_imgs)
            num_pads = seq_len - num_imgs
            indices = np.concatenate(
                [indices, np.ones(num_pads).astype(np.int32) * (num_imgs - 1)]
            )
        assert len(indices) == seq_len
        return indices
