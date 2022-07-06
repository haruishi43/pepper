#!/usr/bin/env python3

import torch

from mmcv.runner import auto_fp16

from ..builder import (
    REID,
    build_backbone,
    build_head,
    build_neck,
    build_temporal_layer,
)
from .base import BaseReID


@REID.register_module()
class VideoReID(BaseReID):

    _stage = ("backbone", "neck", "temporal", "pre_logits")
    _temporal_backbone = False

    def __init__(
        self,
        backbone,
        neck=None,
        temporal=None,
        head=None,
        pretrained=None,
        train_cfg=None,
        init_cfg=None,
        inference_stage=None,
    ):
        super(VideoReID, self).__init__(init_cfg)

        # backbone -> pool (neck) -> temporal -> head

        if pretrained is not None:
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if temporal is not None:
            self.temporal = build_temporal_layer(temporal)
        else:
            # we assume that the backbone takes care of the temporal features
            self._temporal_backbone = True

        if head is not None:
            self.head = build_head(head)

        if inference_stage is not None:
            assert (
                inference_stage in self._stage
            ), f"ERR: {inference_stage} needs one of {self._stage}"
        self.overwrite_stage = inference_stage

    @property
    def with_temporal(self):
        return hasattr(self, "temporal") and self.temporal is not None

    def extract_feat(self, img, stage="pre_logits"):
        assert (
            stage in self._stage
        ), f'Invalid output stage "{stage}", please choose from {self._stage}'

        if img.ndim == 4:
            # force dim=5
            img = img.unsqueeze(0)

        if not self._temporal_backbone:
            b, s, c, h, w = img.shape
            img = img.view(b * s, c, h, w)
            x = self.backbone(img)  # output is a tuple
        else:
            # TODO: might be different if we're using 3D Conv, but that should be another
            # Model (this model is sequential)
            x = self.backbone(img)
        if stage == "backbone":
            return x

        if self.with_neck:
            x = self.neck(x)
            if not self._temporal_backbone:
                if isinstance(x, tuple):
                    x = tuple([_x.view(b, s, -1) for _x in x])
                else:
                    x = x.view(b, s, -1)  # unravel features
        if stage == "neck":
            return x

        if self.with_temporal:
            # we assume tensor shape with (b, s, feat)
            x = self.temporal(x)
        if stage == "temporal":
            return x

        if self.with_head and hasattr(self.head, "pre_logits"):
            x = self.head.pre_logits(x)
        return x

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        assert isinstance(
            img, torch.Tensor
        ), f"ERR: input img should be a tensor, but got {type(img)}"
        assert (
            img.ndim == 5
        ), f"ERR: input img should be 5dim, but got {img.ndim}"

        # 1. compute features
        x = self.extract_feat(img)
        head_outputs = self.head.forward_train(x)

        # 2. compute losses
        losses = dict()
        reid_loss = self.head.loss(gt_label, *head_outputs)

        losses.update(reid_loss)

        return losses

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def simple_test(self, img, **kwargs):
        """Test without augmentation."""

        if isinstance(img, list):
            # for inference, we can have a list of tensors
            num_imgs = len(img)
            img = torch.stack(img, dim=0)  # stack at batch direction
        elif isinstance(img, torch.Tensor):
            if img.ndim == 3:
                # single image
                img = img.unsqueeze(0)
            elif img.ndim == 4:
                # sequence of images
                num_imgs = img.shape[0]
            else:
                raise ValueError(f"{img.shape} is not supported")
        else:
            raise ValueError(f"{type(img)} is not supported")

        if self.overwrite_stage is not None:
            stage = self.overwrite_stage
            feats = self.extract_feat(img, stage=stage)
        else:
            feats = self.extract_feat(img)

        # TODO: might need to post process if feature size is not what we want
        # for example, the number of features are not 1, but `num_imgs`

        # FIXME: handle mutliple outputs
        # mainly features from various levels in the backbone
        if isinstance(feats, tuple):
            if len(feats) > 1:
                # if features from multiple layers are returned, we use the last feature
                feats = feats[-1]
            else:
                feats = feats[0]
        return feats
