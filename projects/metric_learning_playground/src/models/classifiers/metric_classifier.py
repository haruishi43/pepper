#!/usr/bin/env python3

from mmcls.models.builder import (
    CLASSIFIERS,
    build_backbone,
    build_head,
    build_neck,
)
from mmcls.models.utils.augment import Augments
from mmcls.models.classifiers.base import BaseClassifier


@CLASSIFIERS.register_module()
class MetricImageClassifier(BaseClassifier):
    def __init__(
        self,
        backbone,
        neck=None,
        head=None,
        pretrained=None,
        train_cfg=None,
        init_cfg=None,
    ):
        super(MetricImageClassifier, self).__init__(init_cfg)

        if pretrained is not None:
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get("augments", None)
            if augments_cfg is not None:
                self.augments = Augments(augments_cfg)

    def extract_feat(self, img, gt_label=None, stage="pre_logits"):
        assert stage in ["backbone", "neck", "pre_logits"], (
            f'Invalid output stage "{stage}", please choose from "backbone", '
            '"neck" and "pre_logits"'
        )

        x = self.backbone(img)

        if stage == "backbone":
            return x

        if self.with_neck:
            x = self.neck(x)
        if stage == "neck":
            return x

        if self.with_head and hasattr(self.head, "pre_logits"):
            x = self.head.pre_logits(x, gt_label)

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
        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        x = self.extract_feat(img, gt_label)

        losses = dict()
        loss = self.head.forward_train(x, gt_label)
        losses.update(loss)

        return losses

    def simple_test(
        self,
        img,
        img_metas=None,
        return_feats=False,
        **kwargs,
    ):
        """Test without augmentation."""
        x = self.extract_feat(img, gt_label=None)

        # TODO: might want to add options for extracting features
        res = self.head.simple_test(x, return_feats=return_feats, **kwargs)

        return res
