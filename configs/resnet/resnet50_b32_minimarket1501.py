_base_ = [
    "../_base_/datasets/market1501.py",
    "../_base_/schedules/basic_schedule.py",
    "../_base_/default_runtime.py",
]
model = dict(
    reid=dict(
        type="BaseReID",
        backbone=dict(
            type="ResNet",
            depth=50,
            num_stages=4,
            out_indices=(3,),
            style="pytorch",
        ),
        neck=dict(
            type="GlobalAveragePooling",
            kernel_size=(8, 4),
            stride=1,
        ),
        head=dict(
            type="LinearReIDHead",
            num_fcs=1,
            in_channels=2048,
            fc_channels=1024,
            out_channels=128,
            num_classes=380,
            loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
            # loss_pairwise=dict(type="TripletLoss", margin=0.3, loss_weight=1.0),
            norm_cfg=dict(type="BN1d"),
            act_cfg=dict(type="ReLU"),
        ),
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth",  # noqa: E251  # noqa: E501
        ),
    )
)
# data = dict(
#     sampler=dict(
#         type="NaiveIdentityDistributedSampler",
#         batch_size=16,
#         num_instances=4,
#         shuffle=True,
#     ),
# )
data = dict(
    sampler=dict(
        type="BalancedIdentityDistributedSampler",
        batch_size=16,
        num_instances=4,
        shuffle=True,
    ),
)
