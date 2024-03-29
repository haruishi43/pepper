#!/usr/bin/env python3

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
from mmcv.utils import Registry

MODELS = Registry("models", parent=MMCV_MODELS)

BACKBONES = MODELS
TEMPORAL = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS

# Module includes Backbone, neck, heads, and losses
REID = MODELS

ATTENTION = Registry("attention", parent=MMCV_ATTENTION)

METRIC_LINEAR_LAYERS = Registry("linear layer")


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_temporal_layer(cfg):
    """Build temporal modeling layer"""
    return TEMPORAL.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_reid(cfg):
    return REID.build(cfg)
