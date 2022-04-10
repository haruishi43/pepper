# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[5],
)

# runner settings
runner = dict(type="EpochBasedRunner", max_epochs=6)
