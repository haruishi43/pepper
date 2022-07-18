_base_ = [
    "../_base_/models/basic_resnet50.py",
    "../_base_/samplers/default_sampler.py",
    "../_base_/datasets/market1501_REA.py",
    "../_base_/schedules/extended_default_schedule.py",
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
    warmup_ratio=0.01,
)
