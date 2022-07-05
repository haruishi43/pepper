model = dict(
    type="ImageReID",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(3,),
        style="pytorch",
    ),
    neck=dict(type="KernelGlobalAveragePooling", kernel_size=(8, 4), stride=1),
    head=dict(
        type="LinearReIDHead",
        num_fcs=1,
        in_channels=2048,
        fc_channels=1024,
        # out_channels=128,
        out_channels=2048,  # match BoT Baseline
        num_classes=380,
        # loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        loss=dict(
            type="LabelSmoothLoss", label_smooth_val=0.1, loss_weight=1.0
        ),
        loss_pairwise=dict(type="TripletLoss", margin=0.3, loss_weight=0.3),
        norm_cfg=dict(type="BN1d"),
        act_cfg=dict(type="ReLU"),
    ),
    init_cfg=dict(
        type="Pretrained",
        checkpoint="https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth",  # noqa: E251  # noqa: E501
    ),
    inference_stage="pre_logits",
)
sampler = dict(
    type="InfiniteBalancedIdentityDistributedSampler",
    batch_size=32,
    num_instances=4,
    shuffle=True,
)
