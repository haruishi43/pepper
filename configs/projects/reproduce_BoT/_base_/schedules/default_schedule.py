# optimizer
optimizer = dict(type="Adam", lr=0.00035, weight_decay=5e-04, betas=(0.9, 0.99))
optimizer_config = dict(grad_clip=dict(max_norm=5.0, norm_type=2))

# learning policy
# configs for baseline
lr_config = dict(
    # -> policy config
    policy="step",
    step=[2000, 3500, 6000],
    gamma=0.1,
)

# runner settings
runner = dict(type="IterBasedRunner", max_iters=10000)

# evaluation
evaluation = dict(
    interval=2500,
    gpu_collect=True,
    metric=["metric", "CMC", "mAP"],
    normalize_features=False,
    dist_metric="cosine",
    use_metric_cuhk03=False,
    rerank=False,
)

# checkpoint
checkpoint_config = dict(interval=2500)
