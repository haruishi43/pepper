_base_ = [
    "../_base_/models/lenetplusplus.py",
    "../_base_/datasets/mnist_orig.py",
    "../_base_/schedules/default_schedule.py",
    "../_base_/default_runtime.py",
]

model = dict(
    head=dict(
        type="VisualizeFeatureHead",
        vis_dim=2,
        loss_pairwise=[
            # dict(type='TripletLoss', margin=0.3, loss_weight=1.0),
            dict(type="CenterLoss", num_classes=10, feat_dim=2),
        ],
    )
)

optimizer = dict(
    type='SGD',
    lr=0.005,  # 0.01
    momentum=0.9,
    weight_decay=5e-4,
    paramwise_cfg={  # center-loss has parameters which needs to be scheduled differently
        "head.loss_pairwise.0.centers": dict(lr_mult=10, decay_mult=0.0),
    },
)  # with triplet

# learning policy
lr_config = dict(
    policy='step',
    step=[2000],
    gamma=0.1,
    warmup="linear",
    warmup_iters=1000,
    warmup_ratio=0.01,
)

work_dir = './work_dirs/center_loss_mnist/'
