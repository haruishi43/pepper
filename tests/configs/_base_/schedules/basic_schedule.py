# optimizer
optimizer = dict(type="Adam", lr=0.00035, weight_decay=5e-04, betas=(0.9, 0.99))
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy="step",
    step=[500],
    warmup="linear",
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
)

# runner settings
runner = dict(type="IterBasedRunner", max_iters=1000)

# evaluation config
evaluation = dict(
    interval=500,
    gpu_collect=True,
    metric=["metric", "CMC", "mAP"],
    normalize_features=False,
    dist_metric="euclidean",
    use_metric_cuhk03=False,
    rerank=False,
)
