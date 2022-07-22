_base_ = [
    "../_base_/models/lenetplusplus.py",
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

dataset_type = 'MNIST'
img_norm_cfg = dict(mean=[33.46], std=[78.87], to_rgb=True)

train_pipeline = [
    dict(type='Resize', size=28),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label']),
]
test_pipeline = [
    dict(type='Resize', size=28),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type, data_prefix='data/mnist', pipeline=train_pipeline,),
    val=dict(
        type=dataset_type, data_prefix='data/mnist', pipeline=test_pipeline,),
    test=dict(
        type=dataset_type, data_prefix='data/mnist', pipeline=test_pipeline,),
)

evaluation = dict(
    interval=500, metric='accuracy', metric_options={'topk': (1, )}
)

# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)  # no triplet
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    paramwise_cfg={  # center-loss has parameters which needs to be scheduled differently
        "head.loss_pairwise.0.centers": dict(lr_mult=10, decay_mult=0.0),
    },
)  # with triplet

optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    step=[2000],
    gamma=0.1,
    warmup="linear",
    warmup_iters=1000,
    warmup_ratio=0.01,
)

# checkpoint saving
checkpoint_config = dict(interval=2000)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=6000)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/lenet_mnist_2/'
load_from = None
resume_from = None
workflow = [('train', 1)]
