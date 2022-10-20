num_parts = 4
in_channels = 2048
model = dict(
    type="ImageReID",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        strides=(1, 2, 2, 1),  # increases final features by x2
        out_indices=(3,),
        style="pytorch",
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth",  # noqa: E251  # noqa: E501
            prefix="backbone.",
        ),
    ),
    neck=dict(type="PartPooling", num_parts=num_parts),
    head=dict(
        type="PCBHead",
        num_parts=num_parts,
        in_channels=in_channels,
        mid_channels=256,
        num_classes=380,
        loss_cls=dict(
            type="LabelSmoothLoss", label_smooth_val=0.1, loss_weight=1.0 / num_parts
        ),
        norm_cfg=dict(type="BN1d"),
        act_cfg=dict(type="ReLU"),
    ),
    inference_stage="pre_logits",
)
