#!/usr/bin/env python3

import argparse

import torch
from mmcv import Config
from mmcv.cnn.utils import revert_sync_batchnorm

from pepper.models.reid.image_reid import ImageReID


def main(mode, use_gpu=True, train_mode=True):

    if mode == "bot":
        cfg = Config.fromfile(
            "configs/image/bot/bot_resnet50_b64_market1501.py"
        )
        h, w = 256, 128
    elif mode == "mgn":
        cfg = Config.fromfile(
            "configs/image/mgn/mgn_resnet50_b64_market1501.py"
        )
        h, w = 384, 128
    elif mode == "amgn":
        cfg = Config.fromfile(
            "configs/image/amgn/amgn_resnet50_b64_market1501.py"
        )
        h, w = 384, 128
    else:
        raise ValueError(f"{mode} is not a valid mode")
    print(cfg.pretty_text)

    # initialize sample data
    bs = 32
    instances = 4
    img = torch.rand((bs, 3, h, w), dtype=torch.float)
    gt_label = torch.tensor([i % instances for i in range(bs)])

    # initialize model
    net = ImageReID(
        backbone=cfg.model.backbone,
        neck=cfg.model.neck,
        head=cfg.model.head,
        init_cfg=cfg.model.get("init_cfg", None),
    )
    net = revert_sync_batchnorm(net)
    print("num classes:", net.head.num_classes)

    if use_gpu:
        net = net.cuda()
        img = img.cuda()
        gt_label = gt_label.cuda()

    if not train_mode:
        net = net.eval()

    # run model (reuturns loss for training mode)
    out = net(
        img=img,
        gt_label=gt_label,
        return_loss=train_mode,
    )

    if train_mode:
        # losses and metrics (dict)
        print(out.keys())
        print(out)
    else:
        # features (torch.tensor)
        print(out)
        print(out.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    use_gpu = not args.cpu
    train_mode = not args.test

    main(mode=args.mode, use_gpu=use_gpu, train_mode=train_mode)
