# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)  # no triplet
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    step=[2000],
    gamma=0.1,
    # warmup="linear",
    # warmup_iters=1000,
    # warmup_ratio=0.01,
)
