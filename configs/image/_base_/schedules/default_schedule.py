# optimizer
optimizer = dict(type="Adam", lr=0.00035, weight_decay=5e-04, betas=(0.9, 0.99))
optimizer_config = dict(grad_clip=dict(max_norm=5.0, norm_type=2))
# optimizer_config = dict(grad_clip=None)

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

# lr_config = dict(
#     # -> policy config
#     policy="step",
#     step=[2000, 4000],  # [5]
#     gamma=0.1,
#     # -> warmup config
#     warmup="linear",
#     warmup_iters=1000,
#     warmup_ratio=0.1,
# )

# learning rate configs (added warmup by default)
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

# evaluation (euclidean distance by default)
evaluation = dict(
    interval=2500,
    gpu_collect=True,
    metric=["metric", "CMC", "mAP"],
    dist_metric="cosine",  # "euclidean",
    use_metric_cuhk03=False,
    rerank=False,
)

# checkpoint
checkpoint_config = dict(interval=2500)
