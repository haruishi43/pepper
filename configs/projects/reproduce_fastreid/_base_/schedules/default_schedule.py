# optimizer
# optimizer = dict(type="Adam", lr=0.00035, weight_decay=5e-04, betas=(0.9, 0.999))
optimizer = dict(type="Adam", lr=3.5e-06, weight_decay=5e-04, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=dict(max_norm=5.0, norm_type=2))  # clipping not in baseline
# optimizer_config = dict(grad_clip=dict(max_norm=35.0, norm_type=2))  # clipping not in baseline
# optimizer_config = dict(grad_clip=None)

# learning policy
# configs for baseline
lr_config = dict(
    # -> policy config
    policy="step",
    step=[2000, 3500],
    gamma=0.1,
)

# runner settings
runner = dict(type="IterBasedRunner", max_iters=6000)

# evaluation
evaluation = dict(
    interval=2000,
    gpu_collect=True,
    metric=["metric", "CMC", "mAP", "mINP"],
    dist_metric="cosine",
    use_metric_cuhk03=False,
    rerank=False,
)

# checkpoint
checkpoint_config = dict(interval=2000)
