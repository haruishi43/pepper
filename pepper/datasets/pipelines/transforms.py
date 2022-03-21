#!/usr/bin/env python3

import inspect
import math
import random
from numbers import Number
from typing import Sequence

import cv2
import mmcv
import numpy as np

from pepper.core.utils import crop_image

from ..builder import PIPELINES
from .compose import Compose

try:
    import albumentations
except ImportError:
    albumentations = None


@PIPELINES.register_module()
class RandomCrop(object):
    """Crop the given Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. If a sequence of length 4 is provided, it is used to
            pad left, top, right, bottom borders respectively.  If a sequence
            of length 2 is provided, it is used to pad left/right, top/bottom
            borders, respectively. Default: None, which means no padding.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
            Default: False.
        pad_val (Number | Sequence[Number]): Pixel pad_val value for constant
            fill. If a tuple of length 3, it is used to pad_val R, G, B
            channels respectively. Default: 0.
        padding_mode (str): Type of padding. Defaults to "constant". Should
            be one of the following:
            - constant: Pads with a constant value, this value is specified \
                with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: Pads with reflection of image without repeating the \
                last value on the edge. For example, padding [1, 2, 3, 4] \
                with 2 elements on both sides in reflect mode will result \
                in [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: Pads with reflection of image repeating the last \
                value on the edge. For example, padding [1, 2, 3, 4] with \
                2 elements on both sides in symmetric mode will result in \
                [2, 1, 1, 2, 3, 4, 4, 3].
    """

    def __init__(
        self,
        size,
        padding=None,
        pad_if_needed=False,
        pad_val=0,
        padding_mode="constant",
    ):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        # check padding mode
        assert padding_mode in ["constant", "edge", "reflect", "symmetric"]
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.pad_val = pad_val
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: Params (xmin, ymin, target_height, target_width) to be
                passed to ``crop`` for random crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        target_height, target_width = output_size
        if width == target_width and height == target_height:
            return 0, 0, height, width

        ymin = random.randint(0, height - target_height)
        xmin = random.randint(0, width - target_width)
        return ymin, xmin, target_height, target_width

    def __call__(self, results):
        """
        Args:
            img (ndarray): Image to be cropped.
        """
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            if self.padding is not None:
                img = mmcv.impad(
                    img, padding=self.padding, pad_val=self.pad_val
                )

            # pad the height if needed
            if self.pad_if_needed and img.shape[0] < self.size[0]:
                img = mmcv.impad(
                    img,
                    padding=(
                        0,
                        self.size[0] - img.shape[0],
                        0,
                        self.size[0] - img.shape[0],
                    ),
                    pad_val=self.pad_val,
                    padding_mode=self.padding_mode,
                )

            # pad the width if needed
            if self.pad_if_needed and img.shape[1] < self.size[1]:
                img = mmcv.impad(
                    img,
                    padding=(
                        self.size[1] - img.shape[1],
                        0,
                        self.size[1] - img.shape[1],
                        0,
                    ),
                    pad_val=self.pad_val,
                    padding_mode=self.padding_mode,
                )

            ymin, xmin, height, width = self.get_params(img, self.size)
            results[key] = mmcv.imcrop(
                img,
                np.array(
                    [
                        xmin,
                        ymin,
                        xmin + width - 1,
                        ymin + height - 1,
                    ]
                ),
            )
        return results

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(size={self.size}, padding={self.padding})"
        )


@PIPELINES.register_module()
class SeqRandomCrop(RandomCrop):
    """Sequentially random crop the images.
    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped results are generated.
    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            height and width.
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Default False.
        share_params (bool, optional): Whether share the cropping parameters
            for the images.
    """

    def __init__(self, share_params=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_params = share_params

    def __call__(self, results):
        """Call function to sequentially randomly crop images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
            updated according to crop size.
        """
        if self.share_params:
            offsets = self.get_params(results[0]["img"], self.size)
        else:
            offsets = None

        outs = []
        for _results in results:
            _results = self.__call__(_results, offsets)
            if _results is None:
                return None
            outs.append(_results)

        return outs


@PIPELINES.register_module()
class RandomResizedCrop(object):
    """Crop the given image to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size.
    Args:
        size (sequence | int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        scale (tuple): Range of the random size of the cropped image compared
            to the original image. Defaults to (0.08, 1.0).
        ratio (tuple): Range of the random aspect ratio of the cropped image
            compared to the original image. Defaults to (3. / 4., 4. / 3.).
        max_attempts (int): Maximum number of attempts before falling back to
            Central Crop. Defaults to 10.
        efficientnet_style (bool): Whether to use efficientnet style Random
            ResizedCrop. Defaults to False.
        min_covered (Number): Minimum ratio of the cropped area to the original
             area. Only valid if efficientnet_style is true. Defaults to 0.1.
        crop_padding (int): The crop padding parameter in efficientnet style
            center crop. Only valid if efficientnet_style is true.
            Defaults to 32.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to
            'bilinear'.
        backend (str): The image resize backend type, accepted values are
            `cv2` and `pillow`. Defaults to `cv2`.
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        max_attempts=10,
        efficientnet_style=False,
        min_covered=0.1,
        crop_padding=32,
        interpolation="bilinear",
        backend="cv2",
    ):
        if efficientnet_style:
            assert isinstance(size, int)
            self.size = (size, size)
            assert crop_padding >= 0
        else:
            if isinstance(size, (tuple, list)):
                self.size = size
            else:
                self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            raise ValueError(
                "range should be of kind (min, max). "
                f"But received scale {scale} and rato {ratio}."
            )
        assert min_covered >= 0, "min_covered should be no less than 0."
        assert (
            isinstance(max_attempts, int) and max_attempts >= 0
        ), "max_attempts mush be int and no less than 0."
        assert interpolation in (
            "nearest",
            "bilinear",
            "bicubic",
            "area",
            "lanczos",
        )
        if backend not in ["cv2", "pillow"]:
            raise ValueError(
                f"backend: {backend} is not supported for resize."
                'Supported backends are "cv2", "pillow"'
            )

        self.scale = scale
        self.ratio = ratio
        self.max_attempts = max_attempts
        self.efficientnet_style = efficientnet_style
        self.min_covered = min_covered
        self.crop_padding = crop_padding
        self.interpolation = interpolation
        self.backend = backend

    @staticmethod
    def get_params(img, scale, ratio, max_attempts=10):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (ndarray): Image to be cropped.
            scale (tuple): Range of the random size of the cropped image
                compared to the original image size.
            ratio (tuple): Range of the random aspect ratio of the cropped
                image compared to the original image area.
            max_attempts (int): Maximum number of attempts before falling back
                to central crop. Defaults to 10.
        Returns:
            tuple: Params (ymin, xmin, ymax, xmax) to be passed to `crop` for
                a random sized crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        area = height * width

        for _ in range(max_attempts):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            target_width = int(round(math.sqrt(target_area * aspect_ratio)))
            target_height = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_width <= width and 0 < target_height <= height:
                ymin = random.randint(0, height - target_height)
                xmin = random.randint(0, width - target_width)
                ymax = ymin + target_height - 1
                xmax = xmin + target_width - 1
                return ymin, xmin, ymax, xmax

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            target_width = width
            target_height = int(round(target_width / min(ratio)))
        elif in_ratio > max(ratio):
            target_height = height
            target_width = int(round(target_height * max(ratio)))
        else:  # whole image
            target_width = width
            target_height = height
        ymin = (height - target_height) // 2
        xmin = (width - target_width) // 2
        ymax = ymin + target_height - 1
        xmax = xmin + target_width - 1
        return ymin, xmin, ymax, xmax

    # https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/data.py # noqa
    @staticmethod
    def get_params_efficientnet_style(
        img,
        size,
        scale,
        ratio,
        max_attempts=10,
        min_covered=0.1,
        crop_padding=32,
    ):
        """Get parameters for ``crop`` for a random sized crop in efficientnet
        style.
        Args:
            img (ndarray): Image to be cropped.
            size (sequence): Desired output size of the crop.
            scale (tuple): Range of the random size of the cropped image
                compared to the original image size.
            ratio (tuple): Range of the random aspect ratio of the cropped
                image compared to the original image area.
            max_attempts (int): Maximum number of attempts before falling back
                to central crop. Defaults to 10.
            min_covered (Number): Minimum ratio of the cropped area to the
                original area. Only valid if efficientnet_style is true.
                Defaults to 0.1.
            crop_padding (int): The crop padding parameter in efficientnet
                style center crop. Defaults to 32.
        Returns:
            tuple: Params (ymin, xmin, ymax, xmax) to be passed to `crop` for
                a random sized crop.
        """
        height, width = img.shape[:2]
        area = height * width
        min_target_area = scale[0] * area
        max_target_area = scale[1] * area

        for _ in range(max_attempts):
            aspect_ratio = random.uniform(*ratio)
            min_target_height = int(
                round(math.sqrt(min_target_area / aspect_ratio))
            )
            max_target_height = int(
                round(math.sqrt(max_target_area / aspect_ratio))
            )

            if max_target_height * aspect_ratio > width:
                max_target_height = int((width + 0.5 - 1e-7) / aspect_ratio)
                if max_target_height * aspect_ratio > width:
                    max_target_height -= 1

            max_target_height = min(max_target_height, height)
            min_target_height = min(max_target_height, min_target_height)

            # slightly differs from tf implementation
            target_height = int(
                round(random.uniform(min_target_height, max_target_height))
            )
            target_width = int(round(target_height * aspect_ratio))
            target_area = target_height * target_width

            # slight differs from tf. In tf, if target_area > max_target_area,
            # area will be recalculated
            if (
                target_area < min_target_area
                or target_area > max_target_area
                or target_width > width
                or target_height > height
                or target_area < min_covered * area
            ):
                continue

            ymin = random.randint(0, height - target_height)
            xmin = random.randint(0, width - target_width)
            ymax = ymin + target_height - 1
            xmax = xmin + target_width - 1

            return ymin, xmin, ymax, xmax

        # Fallback to central crop
        img_short = min(height, width)
        crop_size = size[0] / (size[0] + crop_padding) * img_short

        ymin = max(0, int(round((height - crop_size) / 2.0)))
        xmin = max(0, int(round((width - crop_size) / 2.0)))
        ymax = min(height, ymin + crop_size) - 1
        xmax = min(width, xmin + crop_size) - 1

        return ymin, xmin, ymax, xmax

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            if self.efficientnet_style:
                get_params_func = self.get_params_efficientnet_style
                get_params_args = dict(
                    img=img,
                    size=self.size,
                    scale=self.scale,
                    ratio=self.ratio,
                    max_attempts=self.max_attempts,
                    min_covered=self.min_covered,
                    crop_padding=self.crop_padding,
                )
            else:
                get_params_func = self.get_params
                get_params_args = dict(
                    img=img,
                    scale=self.scale,
                    ratio=self.ratio,
                    max_attempts=self.max_attempts,
                )
            ymin, xmin, ymax, xmax = get_params_func(**get_params_args)
            img = mmcv.imcrop(img, bboxes=np.array([xmin, ymin, xmax, ymax]))
            results[key] = mmcv.imresize(
                img,
                tuple(self.size[::-1]),
                interpolation=self.interpolation,
                backend=self.backend,
            )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f"(size={self.size}"
        repr_str += f", scale={tuple(round(s, 4) for s in self.scale)}"
        repr_str += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        repr_str += f", max_attempts={self.max_attempts}"
        repr_str += f", efficientnet_style={self.efficientnet_style}"
        repr_str += f", min_covered={self.min_covered}"
        repr_str += f", crop_padding={self.crop_padding}"
        repr_str += f", interpolation={self.interpolation}"
        repr_str += f", backend={self.backend})"
        return repr_str


@PIPELINES.register_module()
class RandomGrayscale(object):
    """Randomly convert image to grayscale with a probability of gray_prob.
    Args:
        gray_prob (float): Probability that image should be converted to
            grayscale. Default: 0.1.
    Returns:
        ndarray: Image after randomly grayscale transform.
    Notes:
        - If input image is 1 channel: grayscale version is 1 channel.
        - If input image is 3 channel: grayscale version is 3 channel
          with r == g == b.
    """

    def __init__(self, gray_prob=0.1):
        self.gray_prob = gray_prob

    def _cvt_gray(self, img):

        num_output_channels = img.shape[2]
        if num_output_channels > 1:
            img = mmcv.rgb2gray(img)[:, :, None]
            img = np.dstack([img for _ in range(num_output_channels)])

        return img

    def __call__(self, results):
        """
        Args:
            img (ndarray): Image to be converted to grayscale.
        Returns:
            ndarray: Randomly grayscaled image.
        """
        for key in results.get("img_fields", ["img"]):
            if random.random() < self.gray_prob:
                img = self._cvt_gray(results[key])
                results[key] = img
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(gray_prob={self.gray_prob})"


@PIPELINES.register_module()
class SeqRandomGrayscale(RandomGrayscale):
    """Gray augmention for images."""

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def __call__(self, results):
        """Call function.
        For each dict in results, perform gray augmention for image in the
        dict.
        Args:
            results (list[dict]): List of dict
        Returns:
            list[dict]: List of dict that contains augmented gray image.
        """

        if np.random.random() < self.gray_prob:
            outs = []
            for _results in results:
                for key in _results.get("img_fields", ["img"]):
                    img = self._cvt_gray(_results[key])
                    _results[key] = img
                    outs.append(_results)
            return outs
        else:
            return results


@PIPELINES.register_module()
class RandomFlip(object):
    """Flip the image randomly.
    Flip the image randomly based on flip probaility and flip direction.
    Args:
        flip_prob (float): probability of the image being flipped. Default: 0.5
        direction (str): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __init__(self, flip_prob=0.5, direction="horizontal"):
        assert 0 <= flip_prob <= 1
        assert direction in ["horizontal", "vertical"]
        self.flip_prob = flip_prob
        self.direction = direction

    def __call__(self, results):
        """Call function to flip image.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """
        if "flip" not in results:
            results["flip"] = (
                True if np.random.rand() < self.flip_prob else False
            )
        if "flip_direction" not in results:
            results["flip_direction"] = self.direction

        if results["flip"]:
            # flip image
            for key in results.get("img_fields", ["img"]):
                results[key] = mmcv.imflip(
                    results[key], direction=results["flip_direction"]
                )
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(flip_prob={self.flip_prob})"


@PIPELINES.register_module()
class SeqRandomFlip(RandomFlip):
    """Randomly flip for sequential images.
    Args:
        share_params (bool): If True, share the flip parameters for all images.
            Defaults to True.
    """

    def __init__(self, share_params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_params = share_params

    def __call__(self, results):
        """Call function.
        For each dict in results, call `RandomFlip` to randomly flip image.
        Args:
            results (list[dict]): List of dict
        Returns:
            list[dict]: List of dict that contains flipped results, 'flip',
            'flip_direction' keys are added into the dict.
        """
        if self.share_params:
            flip = True if np.random.rand() < self.flip_prob else False
            for _results in results:
                _results["flip"] = flip
                _results["flip_direction"] = self.direction

        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class RandomErasing(object):
    """Randomly selects a rectangle region in an image and erase pixels.
    Args:
        erase_prob (float): Probability that image will be randomly erased.
            Default: 0.5
        min_area_ratio (float): Minimum erased area / input image area
            Default: 0.02
        max_area_ratio (float): Maximum erased area / input image area
            Default: 0.4
        aspect_range (sequence | float): Aspect ratio range of erased area.
            if float, it will be converted to (aspect_ratio, 1/aspect_ratio)
            Default: (3/10, 10/3)
        mode (str): Fill method in erased area, can be:
            - const (default): All pixels are assign with the same value.
            - rand: each pixel is assigned with a random value in [0, 255]
        fill_color (sequence | Number): Base color filled in erased area.
            Defaults to (128, 128, 128).
        fill_std (sequence | Number, optional): If set and ``mode`` is 'rand',
            fill erased area with random color from normal distribution
            (mean=fill_color, std=fill_std); If not set, fill erased area with
            random color from uniform distribution (0~255). Defaults to None.
    Note:
        See `Random Erasing Data Augmentation
        <https://arxiv.org/pdf/1708.04896.pdf>`_
        This paper provided 4 modes: RE-R, RE-M, RE-0, RE-255, and use RE-M as
        default. The config of these 4 modes are:
        - RE-R: RandomErasing(mode='rand')
        - RE-M: RandomErasing(mode='const', fill_color=(123.67, 116.3, 103.5))
        - RE-0: RandomErasing(mode='const', fill_color=0)
        - RE-255: RandomErasing(mode='const', fill_color=255)
    """

    def __init__(
        self,
        erase_prob=0.5,
        min_area_ratio=0.02,
        max_area_ratio=0.4,
        aspect_range=(3 / 10, 10 / 3),
        mode="const",
        fill_color=(128, 128, 128),
        fill_std=None,
    ):
        assert isinstance(erase_prob, float) and 0.0 <= erase_prob <= 1.0
        assert (
            isinstance(min_area_ratio, float) and 0.0 <= min_area_ratio <= 1.0
        )
        assert (
            isinstance(max_area_ratio, float) and 0.0 <= max_area_ratio <= 1.0
        )
        assert (
            min_area_ratio <= max_area_ratio
        ), "min_area_ratio should be smaller than max_area_ratio"
        if isinstance(aspect_range, float):
            aspect_range = min(aspect_range, 1 / aspect_range)
            aspect_range = (aspect_range, 1 / aspect_range)
        assert (
            isinstance(aspect_range, Sequence)
            and len(aspect_range) == 2
            and all(isinstance(x, float) for x in aspect_range)
        ), "aspect_range should be a float or Sequence with two float."
        assert all(
            x > 0 for x in aspect_range
        ), "aspect_range should be positive."
        assert (
            aspect_range[0] <= aspect_range[1]
        ), "In aspect_range (min, max), min should be smaller than max."
        assert mode in ["const", "rand"]
        if isinstance(fill_color, Number):
            fill_color = [fill_color] * 3
        assert (
            isinstance(fill_color, Sequence)
            and len(fill_color) == 3
            and all(isinstance(x, Number) for x in fill_color)
        ), "fill_color should be a float or Sequence with three int."
        if fill_std is not None:
            if isinstance(fill_std, Number):
                fill_std = [fill_std] * 3
            assert (
                isinstance(fill_std, Sequence)
                and len(fill_std) == 3
                and all(isinstance(x, Number) for x in fill_std)
            ), "fill_std should be a float or Sequence with three int."

        self.erase_prob = erase_prob
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.aspect_range = aspect_range
        self.mode = mode
        self.fill_color = fill_color
        self.fill_std = fill_std

    def _fill_pixels(self, img, top, left, h, w):
        if self.mode == "const":
            patch = np.empty((h, w, 3), dtype=np.uint8)
            patch[:, :] = np.array(self.fill_color, dtype=np.uint8)
        elif self.fill_std is None:
            # Uniform distribution
            patch = np.random.uniform(0, 256, (h, w, 3)).astype(np.uint8)
        else:
            # Normal distribution
            patch = np.random.normal(self.fill_color, self.fill_std, (h, w, 3))
            patch = np.clip(patch.astype(np.int32), 0, 255).astype(np.uint8)

        img[top : top + h, left : left + w] = patch
        return img

    def __call__(self, results):
        """
        Args:
            results (dict): Results dict from pipeline
        Returns:
            dict: Results after the transformation.
        """
        for key in results.get("img_fields", ["img"]):
            if np.random.rand() > self.erase_prob:
                continue
            img = results[key]
            img_h, img_w = img.shape[:2]

            # convert to log aspect to ensure equal probability of aspect ratio
            log_aspect_range = np.log(
                np.array(self.aspect_range, dtype=np.float32)
            )
            aspect_ratio = np.exp(np.random.uniform(*log_aspect_range))
            area = img_h * img_w
            area *= np.random.uniform(self.min_area_ratio, self.max_area_ratio)

            h = min(int(round(np.sqrt(area * aspect_ratio))), img_h)
            w = min(int(round(np.sqrt(area / aspect_ratio))), img_w)
            top = np.random.randint(0, img_h - h) if img_h > h else 0
            left = np.random.randint(0, img_w - w) if img_w > w else 0
            img = self._fill_pixels(img, top, left, h, w)

            results[key] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(erase_prob={self.erase_prob}, "
        repr_str += f"min_area_ratio={self.min_area_ratio}, "
        repr_str += f"max_area_ratio={self.max_area_ratio}, "
        repr_str += f"aspect_range={self.aspect_range}, "
        repr_str += f"mode={self.mode}, "
        repr_str += f"fill_color={self.fill_color}, "
        repr_str += f"fill_std={self.fill_std})"
        return repr_str


@PIPELINES.register_module()
class SeqRandomErasing(RandomErasing):
    """Sequential Random Erasing images."""

    def __init__(self, share_params=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_params = share_params

    def __call__(self, results):
        if self.share_params:
            erase_prob = np.random.rand()
            # NOTE: assumes that all images are the same size (after Resize)
            img = results[0]["img"]
            img_h, img_w = img.shape[:2]
            # convert to log aspect to ensure equal probability of aspect ratio
            log_aspect_range = np.log(
                np.array(self.aspect_range, dtype=np.float32)
            )
            aspect_ratio = np.exp(np.random.uniform(*log_aspect_range))
            area = img_h * img_w
            area *= np.random.uniform(self.min_area_ratio, self.max_area_ratio)

            h = min(int(round(np.sqrt(area * aspect_ratio))), img_h)
            w = min(int(round(np.sqrt(area / aspect_ratio))), img_w)
            top = np.random.randint(0, img_h - h) if img_h > h else 0
            left = np.random.randint(0, img_w - w) if img_w > w else 0

        outs = []
        for _results in results:
            for key in _results.get("img_fields", ["img"]):
                if not self.share_params:
                    erase_prob = np.random.rand()

                if erase_prob > self.erase_prob:
                    continue

                if not self.share_params:
                    img = results[key]
                    img_h, img_w = img.shape[:2]
                    # convert to log aspect to ensure equal probability of aspect ratio
                    log_aspect_range = np.log(
                        np.array(self.aspect_range, dtype=np.float32)
                    )
                    aspect_ratio = np.exp(np.random.uniform(*log_aspect_range))
                    area = img_h * img_w
                    area *= np.random.uniform(
                        self.min_area_ratio, self.max_area_ratio
                    )

                    h = min(int(round(np.sqrt(area * aspect_ratio))), img_h)
                    w = min(int(round(np.sqrt(area / aspect_ratio))), img_w)
                    top = np.random.randint(0, img_h - h) if img_h > h else 0
                    left = np.random.randint(0, img_w - w) if img_w > w else 0

                img = self._fill_pixels(img, top, left, h, w)
                _results[key] = img

            outs.append(_results)

        return outs


@PIPELINES.register_module()
class Pad(object):
    """Pad images.
    Args:
        size (tuple[int] | None): Expected padding size (h, w). Conflicts with
                pad_to_square. Defaults to None.
        pad_to_square (bool): Pad any image to square shape. Defaults to False.
        pad_val (Number | Sequence[Number]): Values to be filled in padding
            areas when padding_mode is 'constant'. Default to 0.
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default to "constant".
    """

    def __init__(
        self, size=None, pad_to_square=False, pad_val=0, padding_mode="constant"
    ):
        assert (size is None) ^ (pad_to_square is False), (
            "Only one of [size, pad_to_square] should be given, "
            f"but get {(size is not None) + (pad_to_square is not False)}"
        )
        self.size = size
        self.pad_to_square = pad_to_square
        self.pad_val = pad_val
        self.padding_mode = padding_mode

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            if self.pad_to_square:
                target_size = tuple(
                    max(img.shape[0], img.shape[1]) for _ in range(2)
                )
            else:
                target_size = self.size
            img = mmcv.impad(
                img,
                shape=target_size,
                pad_val=self.pad_val,
                padding_mode=self.padding_mode,
            )
            results[key] = img
            results["img_shape"] = img.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"(pad_val={self.pad_val}, "
        repr_str += f"padding_mode={self.padding_mode})"
        return repr_str


@PIPELINES.register_module()
class SeqPad(Pad):
    """Pad images."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        """Call function.
        For each dict in results, call the call function of `Pad` to pad image.
        Args:
            results (list[dict]): List of dict
        Returns:
            list[dict]: List of dict that contains padding results,
            'pad_shape', 'pad_fixed_size' and 'pad_size_divisor' keys are
            added into the dict.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class Resize(object):
    """Resize images.
    Args:
        size (int | tuple): Images scales for resizing (h, w).
            When size is int, the default behavior is to resize an image
            to (size, size). When size is tuple and the second value is -1,
            the image will be resized according to adaptive_side. For example,
            when size is 224, the image is resized to 224x224. When size is
            (224, -1) and adaptive_size is "short", the short side is resized
            to 224 and the other side is computed based on the short side,
            maintaining the aspect ratio.
        interpolation (str): Interpolation method. For "cv2" backend, accepted
            values are "nearest", "bilinear", "bicubic", "area", "lanczos". For
            "pillow" backend, accepted values are "nearest", "bilinear",
            "bicubic", "box", "lanczos", "hamming".
            More details can be found in `mmcv.image.geometric`.
        adaptive_side(str): Adaptive resize policy, accepted values are
            "short", "long", "height", "width". Default to "short".
        backend (str): The image resize backend type, accepted values are
            `cv2` and `pillow`. Default: `cv2`.
    """

    def __init__(
        self,
        size,
        interpolation="bilinear",
        adaptive_side="short",
        backend="cv2",
    ):
        assert isinstance(size, int) or (
            isinstance(size, tuple) and len(size) == 2
        )
        assert adaptive_side in {"short", "long", "height", "width"}

        self.adaptive_side = adaptive_side
        self.adaptive_resize = False
        if isinstance(size, int):
            assert size > 0
            size = (size, size)
        else:
            assert size[0] > 0 and (size[1] > 0 or size[1] == -1)
            if size[1] == -1:
                self.adaptive_resize = True
        if backend not in ["cv2", "pillow"]:
            raise ValueError(
                f"backend: {backend} is not supported for resize."
                'Supported backends are "cv2", "pillow"'
            )
        if backend == "cv2":
            assert interpolation in (
                "nearest",
                "bilinear",
                "bicubic",
                "area",
                "lanczos",
            )
        else:
            assert interpolation in (
                "nearest",
                "bilinear",
                "bicubic",
                "box",
                "lanczos",
                "hamming",
            )
        self.size = size
        self.interpolation = interpolation
        self.backend = backend

    def _resize_img(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            ignore_resize = False
            if self.adaptive_resize:
                h, w = img.shape[:2]
                target_size = self.size[0]

                condition_ignore_resize = {
                    "short": min(h, w) == target_size,
                    "long": max(h, w) == target_size,
                    "height": h == target_size,
                    "width": w == target_size,
                }

                if condition_ignore_resize[self.adaptive_side]:
                    ignore_resize = True
                elif any(
                    [
                        self.adaptive_side == "short" and w < h,
                        self.adaptive_side == "long" and w > h,
                        self.adaptive_side == "width",
                    ]
                ):
                    width = target_size
                    height = int(target_size * h / w)
                else:
                    height = target_size
                    width = int(target_size * w / h)
            else:
                height, width = self.size
            if not ignore_resize:
                img = mmcv.imresize(
                    img,
                    size=(width, height),
                    interpolation=self.interpolation,
                    return_scale=False,
                    backend=self.backend,
                )
                results[key] = img
                results["img_shape"] = img.shape

    def __call__(self, results):
        self._resize_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"interpolation={self.interpolation})"
        return repr_str


@PIPELINES.register_module()
class SeqResize(Resize):
    """Resize for Sequential images.
    Args:
        share_params (bool): If True, share the resize parameters for all
            images. Defaults to True.
    """

    def __init__(self, share_params=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_params = share_params

    def __call__(self, results):
        """Call function.
        For each dict in results, call the call function of `Resize` to resize
        image and corresponding annotations.
        Args:
            results (list[dict]): List of dict
        Returns:
            list[dict]: List of dict that contains resized results,
            'img_shape', 'pad_shape', 'scale_factor', 'keep_ratio' keys
            are added into result dict.
        """
        outs = []
        for i, _results in enumerate(results):
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class CenterCrop(object):
    r"""Center crop the image.
    Args:
        crop_size (int | tuple): Expected size after cropping with the format
            of (h, w).
        efficientnet_style (bool): Whether to use efficientnet style center
            crop. Defaults to False.
        crop_padding (int): The crop padding parameter in efficientnet style
            center crop. Only valid if efficientnet style is True. Defaults to
            32.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Only valid if
            ``efficientnet_style`` is True. Defaults to 'bilinear'.
        backend (str): The image resize backend type, accepted values are
            `cv2` and `pillow`. Only valid if efficientnet style is True.
            Defaults to `cv2`.
    Notes:
        - If the image is smaller than the crop size, return the original
          image.
        - If efficientnet_style is set to False, the pipeline would be a simple
          center crop using the crop_size.
        - If efficientnet_style is set to True, the pipeline will be to first
          to perform the center crop with the ``crop_size_`` as:
        .. math::
            \text{crop\_size\_} = \frac{\text{crop\_size}}{\text{crop\_size} +
            \text{crop\_padding}} \times \text{short\_edge}
        And then the pipeline resizes the img to the input crop size.
    """

    def __init__(
        self,
        crop_size,
        efficientnet_style=False,
        crop_padding=32,
        interpolation="bilinear",
        backend="cv2",
    ):
        if efficientnet_style:
            assert isinstance(crop_size, int)
            assert crop_padding >= 0
            assert interpolation in (
                "nearest",
                "bilinear",
                "bicubic",
                "area",
                "lanczos",
            )
            if backend not in ["cv2", "pillow"]:
                raise ValueError(
                    f"backend: {backend} is not supported for "
                    'resize. Supported backends are "cv2", "pillow"'
                )
        else:
            assert isinstance(crop_size, int) or (
                isinstance(crop_size, tuple) and len(crop_size) == 2
            )
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.efficientnet_style = efficientnet_style
        self.crop_padding = crop_padding
        self.interpolation = interpolation
        self.backend = backend

    def __call__(self, results):
        crop_height, crop_width = self.crop_size[0], self.crop_size[1]
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            # img.shape has length 2 for grayscale, length 3 for color
            img_height, img_width = img.shape[:2]

            # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/preprocessing.py#L118 # noqa
            if self.efficientnet_style:
                img_short = min(img_height, img_width)
                crop_height = (
                    crop_height / (crop_height + self.crop_padding) * img_short
                )
                crop_width = (
                    crop_width / (crop_width + self.crop_padding) * img_short
                )

            y1 = max(0, int(round((img_height - crop_height) / 2.0)))
            x1 = max(0, int(round((img_width - crop_width) / 2.0)))
            y2 = min(img_height, y1 + crop_height) - 1
            x2 = min(img_width, x1 + crop_width) - 1

            # crop the image
            img = mmcv.imcrop(img, bboxes=np.array([x1, y1, x2, y2]))

            if self.efficientnet_style:
                img = mmcv.imresize(
                    img,
                    tuple(self.crop_size[::-1]),
                    interpolation=self.interpolation,
                    backend=self.backend,
                )
            img_shape = img.shape
            results[key] = img
        results["img_shape"] = img_shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f"(crop_size={self.crop_size}"
        repr_str += f", efficientnet_style={self.efficientnet_style}"
        repr_str += f", crop_padding={self.crop_padding}"
        repr_str += f", interpolation={self.interpolation}"
        repr_str += f", backend={self.backend})"
        return repr_str


@PIPELINES.register_module()
class Normalize(object):
    """Normalize the image.
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            results[key] = mmcv.imnormalize(
                results[key], self.mean, self.std, self.to_rgb
            )
        results["img_norm_cfg"] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={list(self.mean)}, "
        repr_str += f"std={list(self.std)}, "
        repr_str += f"to_rgb={self.to_rgb})"
        return repr_str


@PIPELINES.register_module()
class SeqNormalize(Normalize):
    """Normalize for Sequential Images."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        """Call function.
        For each dict in results, call the call function of `Normalize` to
        normalize image.
        Args:
            results (list[dict]): List of dict
        Returns:
            list[dict]: List of dict that contains normalized results,
            'img_norm_cfg' key is added into result dict.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float): How much to jitter brightness.
            brightness_factor is chosen uniformly from
            [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast.
            contrast_factor is chosen uniformly from
            [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation.
            saturation_factor is chosen uniformly from
            [max(0, 1 - saturation), 1 + saturation].
    """

    def __init__(self, brightness, contrast, saturation):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, results):
        brightness_factor = random.uniform(0, self.brightness)
        contrast_factor = random.uniform(0, self.contrast)
        saturation_factor = random.uniform(0, self.saturation)
        color_jitter_transforms = [
            dict(
                type="Brightness",
                magnitude=brightness_factor,
                prob=1.0,
                random_negative_prob=0.5,
            ),
            dict(
                type="Contrast",
                magnitude=contrast_factor,
                prob=1.0,
                random_negative_prob=0.5,
            ),
            dict(
                type="ColorTransform",
                magnitude=saturation_factor,
                prob=1.0,
                random_negative_prob=0.5,
            ),
        ]
        random.shuffle(color_jitter_transforms)
        transform = Compose(color_jitter_transforms)
        return transform(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(brightness={self.brightness}, "
        repr_str += f"contrast={self.contrast}, "
        repr_str += f"saturation={self.saturation})"
        return repr_str


@PIPELINES.register_module()
class SeqColorJitter(ColorJitter):
    def __init__(self, share_params=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.share_params = share_params

    def __call__(self, results):
        brightness_factor = random.uniform(0, self.brightness)
        contrast_factor = random.uniform(0, self.contrast)
        saturation_factor = random.uniform(0, self.saturation)
        color_jitter_transforms = [
            dict(
                type="Brightness",
                magnitude=brightness_factor,
                prob=1.0,
                random_negative_prob=0.5,
            ),
            dict(
                type="Contrast",
                magnitude=contrast_factor,
                prob=1.0,
                random_negative_prob=0.5,
            ),
            dict(
                type="ColorTransform",
                magnitude=saturation_factor,
                prob=1.0,
                random_negative_prob=0.5,
            ),
        ]
        random.shuffle(color_jitter_transforms)
        transform = Compose(color_jitter_transforms)

        outs = []
        for i, _results in enumerate(results):
            _results = transform(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class Lighting(object):
    """Adjust images lighting using AlexNet-style PCA jitter.
    Args:
        eigval (list): the eigenvalue of the convariance matrix of pixel
            values, respectively.
        eigvec (list[list]): the eigenvector of the convariance matrix of pixel
            values, respectively.
        alphastd (float): The standard deviation for distribution of alpha.
            Defaults to 0.1
        to_rgb (bool): Whether to convert img to rgb.
    """

    def __init__(self, eigval, eigvec, alphastd=0.1, to_rgb=True):
        assert isinstance(
            eigval, list
        ), f"eigval must be of type list, got {type(eigval)} instead."
        assert isinstance(
            eigvec, list
        ), f"eigvec must be of type list, got {type(eigvec)} instead."
        for vec in eigvec:
            assert isinstance(vec, list) and len(vec) == len(
                eigvec[0]
            ), "eigvec must contains lists with equal length."
        self.eigval = np.array(eigval)
        self.eigvec = np.array(eigvec)
        self.alphastd = alphastd
        self.to_rgb = to_rgb

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            results[key] = mmcv.adjust_lighting(
                img,
                self.eigval,
                self.eigvec,
                alphastd=self.alphastd,
                to_rgb=self.to_rgb,
            )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(eigval={self.eigval.tolist()}, "
        repr_str += f"eigvec={self.eigvec.tolist()}, "
        repr_str += f"alphastd={self.alphastd}, "
        repr_str += f"to_rgb={self.to_rgb})"
        return repr_str


@PIPELINES.register_module()
class Albu(object):
    """Albumentation augmentation.
    Adds custom transformations from Albumentations library.
    Please, visit `https://albumentations.readthedocs.io`
    to get more information.
    An example of ``transforms`` is as followed:
    .. code-block::
        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]
    Args:
        transforms (list[dict]): A list of albu transformations
        keymap (dict): Contains {'input key':'albumentation-style key'}
    """

    def __init__(self, transforms, keymap=None, update_pad_shape=False):
        if albumentations is None:
            raise RuntimeError("albumentations is not installed")
        else:
            from albumentations import Compose

        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape

        self.aug = Compose([self.albu_builder(t) for t in self.transforms])

        if not keymap:
            self.keymap_to_albu = {
                "img": "image",
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.
        It inherits some of :func:`build_from_cfg` logic.
        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and "type" in cfg
        args = cfg.copy()

        obj_type = args.pop("type")
        if mmcv.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError("albumentations is not installed")
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f"type must be a str or valid type, but got {type(obj_type)}"
            )

        if "transforms" in args:
            args["transforms"] = [
                self.albu_builder(transform) for transform in args["transforms"]
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper.
        Renames keys according to keymap provided.
        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """

        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        results = self.aug(**results)

        if "gt_labels" in results:
            if isinstance(results["gt_labels"], list):
                results["gt_labels"] = np.array(results["gt_labels"])
            results["gt_labels"] = results["gt_labels"].astype(np.int64)

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results["pad_shape"] = results["img"].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f"(transforms={self.transforms})"
        return repr_str


@PIPELINES.register_module()
class SeqBrightnessAug(object):
    """Brightness augmention for images.
    Args:
        jitter_range (float): The range of brightness jitter.
            Defaults to 0..
    """

    def __init__(self, jitter_range=0):
        self.jitter_range = jitter_range

    def __call__(self, results):
        """Call function.
        For each dict in results, perform brightness augmention for image in
        the dict.
        Args:
            results (list[dict]): list of dict that from
                :obj:`mmtrack.base_sot_dataset`.
        Returns:
            list[dict]: list of dict that contains augmented image.
        """
        brightness_factor = np.random.uniform(
            max(0, 1 - self.jitter_range), 1 + self.jitter_range
        )
        outs = []
        for _results in results:
            image = _results["img"]
            image = np.dot(image, brightness_factor).clip(0, 255.0)
            _results["img"] = image
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqShiftScaleAug(object):
    """Shift and rescale images and bounding boxes.
    Args:
        target_size (list[int]): list of int denoting exemplar size and search
            size, respectively. Defaults to [127, 255].
        shift (list[int]): list of int denoting the max shift offset. Defaults
            to [4, 64].
        scale (list[float]): list of float denoting the max rescale factor.
            Defaults to [0.05, 0.18].
    """

    def __init__(
        self, target_size=[127, 255], shift=[4, 64], scale=[0.05, 0.18]
    ):
        self.target_size = target_size
        self.shift = shift
        self.scale = scale

    def _shift_scale_aug(self, image, bbox, target_size, shift, scale):
        """Shift and rescale an image and corresponding bounding box.
        Args:
            image (ndarray): of shape (H, W, 3). Typically H and W equal to
                511.
            bbox (ndarray): of shape (4, ) in [x1, y1, x2, y2] format.
            target_size (int): Exemplar size or search size.
            shift (int): The max shift offset.
            scale (float): The max rescale factor.
        Returns:
            tuple(crop_img, bbox): crop_img is a ndarray of shape
            (target_size, target_size, 3), bbox is the corresponding ground
            truth box in [x1, y1, x2, y2] format.
        """
        img_h, img_w = image.shape[:2]

        scale_x = (2 * np.random.random() - 1) * scale + 1
        scale_y = (2 * np.random.random() - 1) * scale + 1
        scale_x = min(scale_x, float(img_w) / target_size)
        scale_y = min(scale_y, float(img_h) / target_size)
        crop_region = np.array(
            [
                img_w // 2 - 0.5 * scale_x * target_size,
                img_h // 2 - 0.5 * scale_y * target_size,
                img_w // 2 + 0.5 * scale_x * target_size,
                img_h // 2 + 0.5 * scale_y * target_size,
            ]
        )

        shift_x = (2 * np.random.random() - 1) * shift
        shift_y = (2 * np.random.random() - 1) * shift
        shift_x = max(-crop_region[0], min(img_w - crop_region[2], shift_x))
        shift_y = max(-crop_region[1], min(img_h - crop_region[3], shift_y))
        shift = np.array([shift_x, shift_y, shift_x, shift_y])
        crop_region += shift

        crop_img = crop_image(image, crop_region, target_size)
        bbox -= np.array(
            [crop_region[0], crop_region[1], crop_region[0], crop_region[1]]
        )
        bbox /= np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
        return crop_img, bbox

    def __call__(self, results):
        """Call function.
        For each dict in results, shift and rescale the image and the bounding
        box in the dict.
        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.
        Returns:
            list[dict]: List of dict that contains cropped image and
            corresponding ground truth box.
        """
        outs = []
        for i, _results in enumerate(results):
            image = _results["img"]
            gt_bbox = _results["gt_bboxes"][0]

            crop_img, crop_bbox = self._shift_scale_aug(
                image,
                gt_bbox,
                self.target_size[i],
                self.shift[i],
                self.scale[i],
            )
            crop_bbox = crop_bbox[None]

            _results["img"] = crop_img
            if "img_shape" in _results:
                _results["img_shape"] = crop_img.shape
            _results["gt_bboxes"] = crop_bbox
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqColorAug(object):
    """Color augmention for images.
    Args:
        prob (list[float]): The probability to perform color augmention for
            each image. Defaults to [1.0, 1.0].
        rgb_var (list[list]]): The values of color augmentaion. Defaults to
            [[-0.55919361, 0.98062831, -0.41940627],
            [1.72091413, 0.19879334, -1.82968581],
            [4.64467907, 4.73710203, 4.88324118]].
    """

    def __init__(
        self,
        prob=[1.0, 1.0],
        rgb_var=[
            [-0.55919361, 0.98062831, -0.41940627],
            [1.72091413, 0.19879334, -1.82968581],
            [4.64467907, 4.73710203, 4.88324118],
        ],
    ):
        self.prob = prob
        self.rgb_var = np.array(rgb_var, dtype=np.float32)

    def __call__(self, results):
        """Call function.
        For each dict in results, perform color augmention for image in the
        dict.
        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.
        Returns:
            list[dict]: List of dict that contains augmented color image.
        """
        outs = []
        for i, _results in enumerate(results):
            image = _results["img"]

            if self.prob[i] > np.random.random():
                offset = np.dot(self.rgb_var, np.random.randn(3, 1))
                # bgr to rgb
                offset = offset[::-1]
                offset = offset.reshape(3)
                image = (image - offset).astype(np.float32)

            _results["img"] = image
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqBlurAug(object):
    """Blur augmention for images.
    Args:
        prob (list[float]): The probability to perform blur augmention for
            each image. Defaults to [0.0, 0.2].
    """

    def __init__(self, prob=[0.0, 0.2]):
        self.prob = prob

    def __call__(self, results):
        """Call function.
        For each dict in results, perform blur augmention for image in the
        dict.
        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.
        Returns:
            list[dict]: List of dict that contains augmented blur image.
        """
        outs = []
        for i, _results in enumerate(results):
            image = _results["img"]

            if self.prob[i] > np.random.random():
                sizes = np.arange(5, 46, 2)
                size = np.random.choice(sizes)
                kernel = np.zeros((size, size))
                c = int(size / 2)
                wx = np.random.random()
                kernel[:, c] += 1.0 / size * wx
                kernel[c, :] += 1.0 / size * (1 - wx)
                image = cv2.filter2D(image, -1, kernel)

            _results["img"] = image
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqPhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        share_params=True,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.share_params = share_params
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def get_params(self):
        """Generate parameters."""
        params = dict()
        # delta
        if np.random.randint(2):
            params["delta"] = np.random.uniform(
                -self.brightness_delta, self.brightness_delta
            )
        else:
            params["delta"] = None
        # mode
        mode = np.random.randint(2)
        params["contrast_first"] = True if mode == 1 else 0
        # alpha
        if np.random.randint(2):
            params["alpha"] = np.random.uniform(
                self.contrast_lower, self.contrast_upper
            )
        else:
            params["alpha"] = None
        # saturation
        if np.random.randint(2):
            params["saturation"] = np.random.uniform(
                self.saturation_lower, self.saturation_upper
            )
        else:
            params["saturation"] = None
        # hue
        if np.random.randint(2):
            params["hue"] = np.random.uniform(-self.hue_delta, self.hue_delta)
        else:
            params["hue"] = None
        # swap
        if np.random.randint(2):
            params["permutation"] = np.random.permutation(3)
        else:
            params["permutation"] = None
        return params

    def photo_metric_distortion(self, results, params=None):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
            params (dict, optional): Pre-defined parameters. Default to None.
        Returns:
            dict: Result dict with images distorted.
        """
        if params is None:
            params = self.get_params()
        results["img_info"]["color_jitter"] = params

        if "img_fields" in results:
            assert results["img_fields"] == [
                "img"
            ], "Only single img_fields is allowed"
        img = results["img"]
        assert img.dtype == np.float32, (
            "PhotoMetricDistortion needs the input image of dtype np.float32,"
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        )
        # random brightness
        if params["delta"] is not None:
            img += params["delta"]

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        if params["contrast_first"]:
            if params["alpha"] is not None:
                img *= params["alpha"]

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if params["saturation"] is not None:
            img[..., 1] *= params["saturation"]

        # random hue
        if params["hue"] is not None:
            img[..., 0] += params["hue"]
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if not params["contrast_first"]:
            if params["alpha"] is not None:
                img *= params["alpha"]

        # randomly swap channels
        if params["permutation"] is not None:
            img = img[..., params["permutation"]]

        results["img"] = img
        return results

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        if self.share_params:
            params = self.get_params()
        else:
            params = None

        outs = []
        for _results in results:
            _results = self.photo_metric_distortion(_results, params)
            outs.append(_results)

        return outs

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str
