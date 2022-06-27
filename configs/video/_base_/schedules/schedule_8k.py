# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy="poly",
    power=0.9,
    min_lr=1e-4,
    by_epoch=False,
    # warmup="linear",
    # warmup_iters=1000,
    # warmup_ratio=1.0 / 1000,
)

# runner settings
# runner = dict(type="EpochBasedRunner", max_epochs=200)
runner = dict(type="IterBasedRunner", max_iters=8000)

# evaluation
evaluation = dict(
    interval=2000, gpu_collect=True, metric=["metric", "CMC", "mAP"]
)

# checkpoint
checkpoint_config = dict(interval=2000)
