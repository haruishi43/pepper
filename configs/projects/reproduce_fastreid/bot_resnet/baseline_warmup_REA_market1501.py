_base_ = [
    "../_base_/models/basic_resnet50.py",
    "../_base_/datasets/market1501_REA.py",
    "../_base_/schedules/default_schedule.py",
    "../_base_/default_runtime.py",
]
# baseline-s + warmup + REA
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
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
)
sampler = dict(
    batch_size=64,
    num_instances=4,
)
