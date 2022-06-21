#!/usr/bin/env python3

import torch

from mmcv import Config

from pepper.models.reid.image_reid import ImageReID


def main():

    cfg = Config.fromfile("configs/resnet/resnet50_b32_market1501.py")
    print(cfg.pretty_text)

    use_gpu = True
    bs = 32
    instances = 4
    h, w = 256, 128
    img = torch.rand((bs, 3, h, w), dtype=torch.float)
    gt_label = torch.tensor([i % instances for i in range(bs)])

    net = ImageReID(
        backbone=cfg.model.backbone,
        neck=cfg.model.neck,
        head=cfg.model.head,
        init_cfg=cfg.model.init_cfg,
    )
    print("num classes:", net.head.num_classes)

    # run model (reuturns loss for training mode)
    train_mode = False
    out = net(
        img=img,
        gt_label=gt_label,
        return_loss=train_mode,
    )

    if train_mode:
        # losses and metrics
        print(out)
    else:
        # features
        print(out)
        print(out.shape)


if __name__ == "__main__":
    main()
