_base_ = [
    "../_base_/models/agw_resnet50.py",
    "../_base_/datasets/market1501_REA.py",
    "../_base_/schedules/bot_schedule.py",
    "../_base_/default_runtime.py",
]
# baseline-s + warmup + REA + LS + stride=1 + BNNeck
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
)
sampler = dict(
    batch_size=64,
    num_instances=4,
)
