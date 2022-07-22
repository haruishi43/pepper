model = dict(
    type='MetricImageClassifier',
    backbone=dict(type='LeNetPlusPlus'),
    neck=None,
    head=dict(
        type='MetricHead',
        in_channels=1152,
        num_classes=10,
        loss_cls=[
            dict(type='CrossEntropyLoss', loss_weight=1.0),
        ],
        # loss_pairwise=[
        #     dict(type='TripletLoss', margin=0.3, loss_weight=1.0),
        # ],
        norm_cfg=dict(type="BN1d"),
        act_cfg=dict(type="ReLU"),
    )
)
