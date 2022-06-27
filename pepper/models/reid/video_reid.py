#!/usr/bin/env python3

from mmcv.runner import auto_fp16

from ..builder import REID, build_backbone, build_head, build_neck
from .base import BaseReID


@REID.register_module()
class VideoReID(BaseReID):

    _stage = ("backbone", "neck", "pre_logits")

    def __init__(
        self,
        backbone,
        neck=None,
        head=None,
        pretrained=None,
        train_cfg=None,
        init_cfg=None,
        inference_stage=None,
    ):
        super(VideoReID, self).__init__(init_cfg)

        if pretrained is not None:
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        # TODO: add temporal necks?

        if head is not None:
            self.head = build_head(head)

        if inference_stage is not None:
            assert (
                inference_stage in self._stage
            ), f"ERR: {inference_stage} needs one of {self._stage}"
        self.overwrite_stage = inference_stage

    def extract_feat(self, img, stage="pre_logits"):
        """Directly extract features from the specified stage.
        Args:
            img (Tensor): The input images. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
            stage (str): Which stage to output the feature. Choose from
                "backbone", "neck" and "pre_logits". Defaults to "pre_logits".
        Returns:
            tuple | Tensor: The output of specified stage.
                The output depends on detailed implementation. In general, the
                output of backbone and neck is a tuple and the output of
                pre_logits is a tensor.

        Examples:
            1. Backbone output
            >>> import torch
            >>> from mmcv import Config
            >>> from pepper.models import build_reid
            >>>
            >>> cfg = Config.fromfile('configs/resnet/resnet18_8xb32_in1k.py').model
            >>> cfg.backbone.out_indices = (0, 1, 2, 3)  # Output multi-scale feature maps
            >>> model = build_reid(cfg)
            >>> outs = model.extract_feat(torch.rand(1, 3, 224, 224), stage='backbone')
            >>> for out in outs:
            ...     print(out.shape)
            torch.Size([1, 64, 56, 56])
            torch.Size([1, 128, 28, 28])
            torch.Size([1, 256, 14, 14])
            torch.Size([1, 512, 7, 7])

            2. Neck output
            >>> import torch
            >>> from mmcv import Config
            >>> from pepper.models import build_reid
            >>>
            >>> cfg = Config.fromfile('configs/resnet/resnet18_8xb32_in1k.py').model
            >>> cfg.backbone.out_indices = (0, 1, 2, 3)  # Output multi-scale feature maps
            >>> model = build_reid(cfg)
            >>>
            >>> outs = model.extract_feat(torch.rand(1, 3, 224, 224), stage='neck')
            >>> for out in outs:
            ...     print(out.shape)
            torch.Size([1, 64])
            torch.Size([1, 128])
            torch.Size([1, 256])
            torch.Size([1, 512])

            3. Pre-logits output (without the final linear classifier head)
            >>> import torch
            >>> from mmcv import Config
            >>> from pepper.models import build_reid
            >>>
            >>> cfg = Config.fromfile('configs/vision_transformer/vit-base-p16_pt-64xb64_in1k-224.py').model
            >>> model = build_reid(cfg)
            >>>
            >>> out = model.extract_feat(torch.rand(1, 3, 224, 224), stage='pre_logits')
            >>> print(out.shape)  # The hidden dims in head is 3072
            torch.Size([1, 3072])
        """  # noqa: E501
        assert (
            stage in self._stage
        ), f'Invalid output stage "{stage}", please choose from {self._stage}'

        x = self.backbone(img)

        if stage == "backbone":
            return x

        if self.with_neck:
            x = self.neck(x)
        if stage == "neck":
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
        if img.nelement() > 0:
            if self.overwrite_stage is not None:
                stage = self.overwrite_stage
                feats = self.extract_feat(img, stage=stage)
            else:
                feats = self.extract_feat(img)

            # FIXME: handle mutliple outputs
            # mainly features from various levels in the backbone
            if isinstance(feats, tuple):
                if len(feats) > 1:
                    # if features from multiple layers are returned, we use the last feature
                    feats = feats[-1]
                else:
                    feats = feats[0]
            return feats
        else:
            return img.new_zeros(0, self.head.out_channels)
