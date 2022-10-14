#!/usr/bin/env python3

import argparse

import torch

from pepper.models import build_backbone


def debug_backbone(mode, use_gpu=False):

    bs = 32
    h, w = (256, 128)

    if mode == "basic":
        # Basic backbone
        # 1/4, 1/8, 1/16, 1/16
        backbone_cfg = dict(
            type="ResNet",
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            style="pytorch",
        )
    elif mode == "bot":
        # BoT backbone
        # 1/4, 1/8, 1/16, 1/16
        backbone_cfg = dict(
            type="ResNet",
            depth=50,
            num_stages=4,
            strides=(1, 2, 2, 1),
            out_indices=(0, 1, 2, 3),
            style="pytorch",
        )
    elif mode == "pcb":
        # PCB backbone
        # 1/4, 1/8, 1/16, 1/16
        backbone_cfg = dict(
            type="ResNet",
            depth=50,
            num_stages=4,
            strides=(1, 2, 2, 1),
            out_indices=(0, 1, 2, 3),
            style="pytorch",
        )
        # need to change the input resolution to support 6 parts
        h, w = (384, 128)
    else:
        raise ValueError(f"mode {mode} is not supported")

    backbone = build_backbone(backbone_cfg)
    img = torch.rand((bs, 3, h, w), dtype=torch.float)
    if use_gpu:
        img = img.cuda()
        backbone = backbone.cuda()

    features = backbone(img)

    for feat in features:
        _h, _w = feat.shape[2:]
        print(feat.shape, f"1/{h/_h}, 1/{w/_w}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    debug_backbone(mode=args.mode, use_gpu=(not args.cpu))
