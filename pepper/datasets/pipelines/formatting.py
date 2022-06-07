#!/usr/bin/env python3

from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from PIL import Image

from ..builder import PIPELINES


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.
    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f"Type {type(data)} cannot be converted to tensor."
            "Supported types are: `numpy.ndarray`, `torch.Tensor`, "
            "`Sequence`, `int` and `float`"
        )


@PIPELINES.register_module()
class ToTensor(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(keys={self.keys})"


@PIPELINES.register_module()
class ImageToTensor(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = to_tensor(img.transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(keys={self.keys})"


@PIPELINES.register_module()
class Transpose(object):
    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def __call__(self, results):
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(keys={self.keys}, order={self.order})"
        )


@PIPELINES.register_module()
class ToPIL(object):
    def __init__(self):
        pass

    def __call__(self, results):
        results["img"] = Image.fromarray(results["img"])
        return results


@PIPELINES.register_module()
class ToNumpy(object):
    def __init__(self):
        pass

    def __call__(self, results):
        results["img"] = np.array(results["img"], dtype=np.float32)
        return results


@PIPELINES.register_module()
class Collect(object):
    """Collect data from the loader relevant to the specific task.
    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img" and "gt_label".
    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'flip',
            'flip_direction', 'img_norm_cfg')
    Returns:
        dict: The result dict contains the following keys
            - keys in ``self.keys``
            - ``img_metas`` if available
    """

    def __init__(
        self,
        keys,
        meta_keys=(
            "filename",
            "ori_filename",
            "ori_shape",
            "img_shape",
            "flip",
            "flip_direction",
            "img_norm_cfg",
            "pid",
            "camid",
            "split",  # FIXME: remove
            "debug_index",  # FIXME: remove
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
            elif key in results["img_info"]:
                img_meta[key] = results["img_info"][key]
        data["img_metas"] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(keys={self.keys}, meta_keys={self.meta_keys})"
        )


@PIPELINES.register_module()
class VideoCollect(object):
    """Collect data from the loader relevant to the specific task.
    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str]): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Defaults to None.
        default_meta_keys (tuple): Default meta keys. Defaults to ('filename',
            'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
            'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
            'frame_id', 'is_video_data').
    """

    def __init__(
        self,
        keys,
        meta_keys=None,
        default_meta_keys=(
            "filename",
            "ori_filename",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "img_norm_cfg",
            "pid",
            "camid",
            "frame_id",
            "is_video_data",
        ),
    ):
        self.keys = keys
        self.meta_keys = default_meta_keys
        if meta_keys is not None:
            if isinstance(meta_keys, str):
                meta_keys = (meta_keys,)
            else:
                assert isinstance(
                    meta_keys, tuple
                ), "meta_keys must be str or tuple"
            self.meta_keys += meta_keys

    def __call__(self, results):
        """Call function to collect keys in results.
        The keys in ``meta_keys`` and ``default_meta_keys`` will be converted
        to :obj:mmcv.DataContainer.
        Args:
            results (list[dict] | dict): List of dict or dict which contains
                the data to collect.
        Returns:
            list[dict] | dict: List of dict or dict that contains the
            following keys:
            - keys in ``self.keys``
            - ``img_metas``
        """
        results_is_dict = isinstance(results, dict)
        if results_is_dict:
            results = [results]
        outs = []
        for _results in results:
            _results = self._add_default_meta_keys(_results)
            _results = self._collect_meta_keys(_results)
            outs.append(_results)

        if results_is_dict:
            outs[0]["img_metas"] = DC(outs[0]["img_metas"], cpu_only=True)

        return outs[0] if results_is_dict else outs

    def _collect_meta_keys(self, results):
        """Collect `self.keys` and `self.meta_keys` from `results` (dict)."""
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
            elif key in results["img_info"]:
                img_meta[key] = results["img_info"][key]
        data["img_metas"] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def _add_default_meta_keys(self, results):
        """Add default meta keys.
        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img = results["img"]
        results.setdefault("pad_shape", img.shape)
        results.setdefault("scale_factor", 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault(
            "img_norm_cfg",
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False,
            ),
        )
        return results


@PIPELINES.register_module()
class FormatBundle(object):
    """Formatting bundle.
    It first concatenates common fields, then simplifies the pipeline of
    formatting common fields, including "img", and "gt_label".
    These fields are formatted as follows.
    - img: (1) transpose, (2) to tensor, (3) to DataContainer (stack=True)
    - gt_labels: (1) to tensor, (2) to DataContainer
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, results):
        """ReID formatting bundle call function.
        Args:
            results (list[dict] or dict): List of dicts or dict.
        Returns:
            dict: The result dict contains the data that is formatted with
            ReID bundle.
        """
        inputs = dict()
        if isinstance(results, list):  # video
            assert len(results) > 1, (
                "the 'results' only have one item, "
                "please directly use normal pipeline not 'Seq' pipeline."
            )
            inputs["img"] = np.stack(
                [_results["img"] for _results in results], axis=3
            )
            inputs["gt_label"] = np.stack(
                [_results["gt_label"] for _results in results], axis=0
            )
            inputs["img_metas"] = [
                _results["img_metas"] for _results in results
            ]
        elif isinstance(results, dict):  # image
            inputs["img"] = results["img"]
            inputs["gt_label"] = results["gt_label"]
            inputs["img_metas"] = results["img_metas"]
        else:
            raise TypeError("results must be a list or a dict.")
        outs = self.reid_format_bundle(inputs)

        return outs

    def reid_format_bundle(self, results):
        """Transform and format gt_label fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
            ReID bundle.
        """
        for key in results:
            if key == "img":
                img = results[key]
                if img.ndim == 3:
                    # image
                    # reorder (c, h, w)
                    img = np.ascontiguousarray(img.transpose(2, 0, 1))
                else:
                    # video
                    # reorder (f, c, h, w)
                    img = np.ascontiguousarray(img.transpose(3, 2, 0, 1))
                results["img"] = DC(to_tensor(img), stack=True)
            elif key == "gt_label":
                results[key] = DC(
                    to_tensor(results[key]), stack=True, pad_dims=None
                )
            elif key == "img_metas":
                continue
            else:
                raise KeyError(f"key {key} is not supported")
        return results


@PIPELINES.register_module()
class WrapFieldsToLists(object):
    """Wrap fields of the data dictionary into lists for evaluation.
    This class can be used as a last step of a test or validation
    pipeline for single image evaluation or inference.
    Example:
        >>> test_pipeline = [
        >>>    dict(type='LoadImageFromFile'),
        >>>    dict(type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
        >>>    dict(type='ImageToTensor', keys=['img']),
        >>>    dict(type='Collect', keys=['img']),
        >>>    dict(type='WrapIntoLists')
        >>> ]
    """

    def __call__(self, results):
        # Wrap dict fields into lists
        for key, val in results.items():
            results[key] = [val]
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}()"
