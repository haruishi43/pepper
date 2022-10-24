#!/usr/bin/env python3

"""Wrapper for ResNet to obtain MGN backbone"""

from copy import deepcopy

import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer

from .base_backbone import BaseBackbone
from .resnet import ResNet, Bottleneck
from ..builder import BACKBONES


@BACKBONES.register_module()
class MGNResNet(BaseBackbone):
    def __init__(
        self,
        depth,
        dilations=(1, 1, 1, 1),
        num_globals=1,
        num_parts=2,
        deep_stem=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        resnet_init_cfg=dict(
            type="Pretrained",
            checkpoint="https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth",  # noqa: E251  # noqa: E501
            prefix="backbone.",
        ),
        init_cfg=None,
    ):
        """
        Args:
            depth: int
                ResNet depth (default=50)
            dilaitons: Tuple(int)
            num_globals: int
                number of global branches (default=1)
            num_parts: int
                number of n-part branches (default=2)
            resnet_init_cfg: dict
                `init_cfg` for `pepper.models.backbones.resnet.ResNet`
        """

        super().__init__(init_cfg)

        assert depth == 50, "only supports resnet50 for now"

        resnet = ResNet(
            depth=depth,
            num_stages=4,
            strides=(1, 2, 2, 2),
            dilations=dilations,
            deep_stem=deep_stem,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            init_cfg=resnet_init_cfg,
        )
        # initialize weights before proceeding
        resnet.init_weights()

        if resnet.deep_stem:
            backbone = nn.Sequential(
                resnet.stem,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3[0],
            )
        else:
            backbone = nn.Sequential(
                resnet.conv1,
                resnet.norm1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3[0],
            )
        self.backbone = backbone

        block = Bottleneck

        # global branch -> stride 2
        # part branches -> stride 1

        # need to wrap in list
        res_conv4 = nn.Sequential(*list(resnet.layer3)[1:])

        res_g_conv5 = resnet.layer4

        downsample = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                1024,
                512 * block.expansion,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            build_norm_layer(norm_cfg, 512 * block.expansion)[1],
        )

        res_p_conv5 = nn.Sequential(
            block(
                inplanes=1024,
                planes=512,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
            ),
            block(
                inplanes=2048,
                planes=512,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
            ),
            block(
                inplanes=2048,
                planes=512,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
            ),
        )
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        parts = []
        for _ in range(num_globals):
            parts.append(
                nn.Sequential(deepcopy(res_conv4), deepcopy(res_g_conv5))
            )
        for _ in range(num_parts):
            parts.append(
                nn.Sequential(deepcopy(res_conv4), deepcopy(res_p_conv5))
            )
        self.parts = nn.ModuleList(parts)

        del resnet, res_conv4, res_g_conv5, res_p_conv5

    def init_weights(self):
        # we don't need to initialize anything
        pass

    def forward(self, x):
        x = self.backbone(x)
        p = []
        for part in self.parts:
            p.append(part(x))
        return tuple(p)
