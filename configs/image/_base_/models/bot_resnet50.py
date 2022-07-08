model = dict(
    type="ImageReID",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        strides=(1, 2, 2, 1),  # increases final features by x2
        out_indices=(3,),
        style="pytorch",
    ),
    # is this different from normal AdaptiveAveragePooling?
    neck=dict(type="KernelGlobalAveragePooling", kernel_size=(16, 8), stride=1),
    head=dict(
        type="BoTReIDHead",
        in_channels=2048,
        num_classes=380,
        loss=dict(
            type="LabelSmoothLoss", label_smooth_val=0.1, loss_weight=1.0
        ),
        loss_pairwise=dict(type="TripletLoss", margin=0.3, loss_weight=1.0),
        norm_cfg=dict(type="BN1d"),
        act_cfg=dict(type="ReLU"),
    ),
    init_cfg=dict(
        type="Pretrained",
        checkpoint="https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth",  # noqa: E251  # noqa: E501
    ),
    # init_cfg=dict(
    #     type="Pretrained",
    #     checkpoint="https://download.pytorch.org/models/resnet50-19c8e357.pth",
    # ),
    inference_stage="pre_logits",
)
sampler = dict(
    type="InfiniteBalancedIdentityDistributedSampler",
    batch_size=32,
    num_instances=4,
    shuffle=True,
)