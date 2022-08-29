model = dict(
    type="ImageReID",
    backbone=dict(
        type="PluginResNet",
        depth=50,
        num_stages=4,
        strides=(1, 2, 2, 1),  # increases final features by x2
        out_indices=(3,),
        style="pytorch",
        plugins=[
            dict(  # 2nd
                cfg=dict(type='NonLocal2d', mode="dot_product"),
                stages=(False, True, False, False),
                layers=(2, 3),
                position='after_conv3',
            ),
            dict(  # 3rd
                cfg=dict(type='NonLocal2d', mode="dot_product"),
                stages=(False, False, True, False),
                layers=(3, 4, 5),
                position='after_conv3',
            ),
        ],
    ),
    neck=dict(type="GeneralizedMeanPooling"),
    head=dict(
        type="BoTHead",
        in_channels=2048,
        num_classes=380,
        loss_cls=dict(
            type="LabelSmoothLoss", label_smooth_val=0.1, loss_weight=1.0
        ),
        loss_pairwise=dict(type="TripletLoss", margin=0.0, hard_mining=False, loss_weight=1.0),
        # loss_circle=dict(type="CircleLoss", margin=0.25, gamma=128, loss_weight=1.0),
        norm_cfg=dict(type="BN1d"),
        act_cfg=dict(type="ReLU"),
    ),
    init_cfg=dict(
        type="Pretrained",
        checkpoint="https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth",  # noqa: E251  # noqa: E501
    ),
    inference_stage="pre_logits",
)
