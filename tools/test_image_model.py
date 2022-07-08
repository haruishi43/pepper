#!/usr/bin/env python3

import torch

from mmcv import Config

from pepper.models.reid.image_reid import ImageReID


def main():

    cfg = Config.fromfile(
        "configs/image/bot_resnet/bot_resnet50_b64_market1501.py"
    )
    print(cfg.pretty_text)

    use_gpu = True
    train_mode = True
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

    if use_gpu:
        net = net.cuda()
        img = img.cuda()
        gt_label = gt_label.cuda()

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
    main()
