model = dict(
    type="ImageReID",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(3,),
        style="pytorch",
    ),
    neck=dict(type="GlobalAveragePooling", dim=2),
    head=dict(
        type="BasicHead",
        in_channels=2048,
        num_classes=380,
        loss_cls=dict(type="CrossEntropyLoss", loss_weight=1.0),
        loss_pairwise=dict(type="TripletLoss", margin=0.3, loss_weight=1.0),
        norm_cfg=dict(type="BN1d"),
        act_cfg=dict(type="ReLU"),
    ),
    init_cfg=dict(
        type="Pretrained",
        checkpoint="https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth",  # noqa: E251  # noqa: E501
    ),
    inference_stage="pre_logits",
)
