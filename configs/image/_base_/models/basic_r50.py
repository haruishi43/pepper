norm_cfg = dict(type="SyncBN", requires_grad=True)  # does this help?
model = dict(
    type="ImageReID",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(3,),
        style="pytorch",
        norm_cfg=norm_cfg,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth",  # noqa: E251  # noqa: E501
            prefix="backbone.",
        ),
    ),
    # neck=dict(type="KernelGlobalAveragePooling", kernel_size=(8, 4), stride=1),
    neck=dict(type="GlobalAveragePooling", dim=2),
    head=dict(
        type="BasicHead",
        in_channels=2048,
        num_classes=380,
        loss_cls=[
            dict(
                type="LabelSmoothLoss", label_smooth_val=0.1, loss_weight=1.0
            ),
        ],
        loss_pairwise=[
            dict(type="TripletLoss", margin=0.3, loss_weight=1.0),
            dict(type="CircleLoss", margin=0.25, gamma=128, loss_weight=1.0 / 64),
        ],
        norm_cfg=dict(type="BN1d", requires_grad=True),
        act_cfg=dict(type="ReLU"),
    ),

    inference_stage="pre_logits",
)
