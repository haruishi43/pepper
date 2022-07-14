#!/usr/bin/env python3

from torch import nn as nn

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import Sequential

from ..builder import BACKBONES
from .resnet import ResNet


class CustomResLayer(Sequential):
    """ResLayer to build ResNet style backbone.
    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(
        self,
        block,
        inplanes,
        planes,
        num_blocks,
        stride=1,
        avg_down=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        downsample_first=True,
        plugins=None,
        **kwargs,
    ):
        self.block = block

        if plugins is not None:
            plugins_per_layer = {}
            assert isinstance(plugins, list)

            for plugin in plugins:
                plug_layers = plugin.pop("layers", None)

                if plug_layers is None:
                    # add plug in if layers are not specified
                    for i in range(num_blocks):
                        if i in plugins_per_layer.keys():
                            plugins_per_layer[i].append(plugin.copy())
                        else:
                            plugins_per_layer[i] = [plugin.copy()]
                    continue

                if isinstance(plug_layers, int):
                    plug_layers = [plug_layers]
                assert isinstance(
                    plug_layers, (list, tuple)
                ), f"{plug_layers} should be list or tuple"

                plug_layers = list(set(plug_layers))  # remove duplicates
                for pl in plug_layers:
                    assert 0 <= pl < num_blocks, f"{pl} is out of range"
                    if pl in plugins_per_layer.keys():
                        plugins_per_layer[pl].append(plugin.copy())
                    else:
                        plugins_per_layer[pl] = [plugin.copy()]
        else:
            plugins_per_layer = {}

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False,
                    )
                )
            downsample.extend(
                [
                    build_conv_layer(
                        conv_cfg,
                        inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=conv_stride,
                        bias=False,
                    ),
                    build_norm_layer(norm_cfg, planes * block.expansion)[1],
                ]
            )
            downsample = nn.Sequential(*downsample)

        plug_idx = 0
        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    plugins=plugins_per_layer.get(plug_idx, None),
                    **kwargs,
                )
            )
            plug_idx += 1
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        plugins=plugins_per_layer.get(plug_idx, None),
                        **kwargs,
                    )
                )
                plug_idx += 1

        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        plugins=plugins_per_layer.get(plug_idx, None),
                        **kwargs,
                    )
                )
                plug_idx += 1
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    plugins=plugins_per_layer.get(plug_idx, None),
                    **kwargs,
                )
            )
        super(CustomResLayer, self).__init__(*layers)


@BACKBONES.register_module()
class BetterPlugResNet(ResNet):
    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage.
        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block`` into the backbone
        like ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.
        An example of plugins format could be:
        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True),
            ...          position='after_conv2'),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='1'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='2'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3')
            ... ]
            >>> self = ResNet(depth=18)
            >>> stage_plugins = self.make_stage_plugins(plugins, 0)
            >>> assert len(stage_plugins) == 3
        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:
        .. code-block:: none
            conv1-> conv2->conv3->yyy->zzz1->zzz2
        Suppose 'stage_idx=1', the structure of blocks in the stage would be:
        .. code-block:: none
            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2
        If stages is missing, the plugin would be applied to all stages.
        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build
        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop("stages", None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return CustomResLayer(**kwargs)
