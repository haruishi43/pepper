# optimizer
optimizer = dict(type="Adam", lr=0.00035, weight_decay=5e-04, betas=(0.9, 0.99))
optimizer_config = dict(type="Fp16OptimizerHook", grad_clip=dict(max_norm=5.0, norm_type=2), loss_scale=512.)

# BoT schedule
lr_config = dict(
    # -> policy config
    policy="step",
    step=[2000, 3500],
    gamma=0.1,
    # -> warmup config
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.01,
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
