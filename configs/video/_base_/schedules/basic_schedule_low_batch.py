# optimizer
# optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type="Adam", lr=0.00035, weight_decay=5e-04, betas=(0.9, 0.99))
optimizer_config = dict(grad_clip=None)

# learning policy
# _wu = 1000  # 1000
# lr_config = dict(
#     # -> policy config
#     policy="step",
#     step=[1000],  # [5]
#     gamma=0.1,
#     # -> warmup config
#     warmup="linear",
#     warmup_iters=_wu,
#     warmup_ratio=1.0 / _wu,
# )

lr_config = dict(
    # -> policy config
    policy="step",
    step=[4000, 8000],  # [5]
    gamma=0.1,
    # -> warmup config
    warmup="linear",
    warmup_iters=1000,
    warmup_ratio=0.1,
)

# runner settings
runner = dict(type="IterBasedRunner", max_iters=16000)

# evaluation
evaluation = dict(
    interval=4000,
    gpu_collect=True,
    metric=["metric", "CMC", "mAP"],
    normalize_features=False,
    dist_metric="euclidean",
    use_metric_cuhk03=False,
    rerank=False,
)

# checkpoint
checkpoint_config = dict(interval=4000)
