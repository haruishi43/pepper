_base_ = [
    "../_base_/models/lenetplusplus.py",
    "../_base_/datasets/mnist_orig.py",
    # "../_base_/schedules/default_schedule.py",
    "../_base_/default_runtime.py",
]

model = dict(
    head=dict(
        type="VisualizeFeatureHead",
        vis_dim=3,
        loss_cls=[dict(type='CrossEntropyLoss', loss_weight=1.0)],
        # loss_pairwise=None,  # only ce
        loss_pairwise=[
            dict(type='CosFace', margin=0.25, gamma=80., loss_weight=1.0),
        ],
        linear_layer=dict(
            type='CosSoftmax',
            scale=30,
            margin=0.5,
        ),
    )
)

# optimizer = dict(
#     type='SGD',
#     lr=0.001,
#     momentum=0.9,
#     weight_decay=5e-4,
#     paramwise_cfg={
#         "head.classifier": dict(lr_mult=1, decay_mult=0.0),
#         "head.fc2": dict(lr_mult=1, decay_mult=0.0),
#     },
# )

optimizer = dict(
    type='Adam',
    lr=3.5e-4,
    weight_decay=5e-4,
    paramwise_cfg={
        "head.classifier": dict(lr_mult=1, decay_mult=0.0),
        "head.fc2": dict(lr_mult=1, decay_mult=0.0),
    },
)

# optimizer_config = dict(grad_clip=None)
optimizer_config = dict(grad_clip=dict(max_norm=5.0, norm_type=2))

lr_config = dict(
    policy='step',
    # step=[500, 1000, 1500, 2000, 4000],
    step=1000,
    gamma=0.1,
    # warmup="linear",
    # warmup_iters=1000,
    # warmup_ratio=0.01,
)

work_dir = './work_dirs/cosface_mnist/'
