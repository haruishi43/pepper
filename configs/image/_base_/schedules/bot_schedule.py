# optimizer
# optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type="Adam", lr=0.00035, weight_decay=5e-04, betas=(0.9, 0.99))
# optimizer_config = dict(grad_clip=None)
optimizer_config = dict(grad_clip=dict(max_norm=5.0, norm_type=2))

# configs based on BoT
lr_config = dict(
    # -> policy config
    policy="step",
    step=[2000, 3500, 6000],
    gamma=0.1,
    # -> warmup config
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
)

# runner settings
runner = dict(type="IterBasedRunner", max_iters=10000)

# evaluation
evaluation = dict(
    interval=2500,
    gpu_collect=True,
    metric=["metric", "CMC", "mAP"],
    normalize_features=False,
    dist_metric="euclidean",
    use_metric_cuhk03=False,
    rerank=False,
)

# checkpoint
checkpoint_config = dict(interval=2500)
