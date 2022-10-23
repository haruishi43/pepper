norm_cfg = dict(type="SyncBN", requires_grad=True)  # does this help?
model = dict(
    type="ImageReID",
    backbone=dict(
        type="MGNResNet",
        depth=50,
        norm_cfg=norm_cfg,
        resnet_init_cfg=dict(
            type="Pretrained",
            checkpoint="https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth",  # noqa: E251  # noqa: E501
            prefix="backbone.",
        ),
    ),
    neck=dict(type="MGNPooling"),
    head=dict(
        type="MGNHead",
        in_channels=2048,
        out_channels=256,
        num_classes=380,
        loss_cls=dict(type="LabelSmoothLoss", label_smooth_val=0.1, loss_weight=1.0 / 8),
        loss_pairwise=[
            dict(type="TripletLoss", margin=0.3, loss_weight=1.0),
        ],
        norm_cfg=dict(type="BN2d", requires_grad=True),
        act_cfg=dict(type="ReLU"),
    ),
    inference_stage="pre_logits",
)
