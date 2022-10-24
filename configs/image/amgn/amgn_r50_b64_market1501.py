_base_ = [
    "../_base_/models/amgn_r50.py",
    "../_base_/samplers/infinite_balanced.py",
    "../_base_/datasets/market1501_384x128.py",
    "../_base_/schedules/bot_schedule.py",
    "../_base_/default_runtime.py",
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
)
sampler = dict(
    batch_size=64,
    num_instances=4,
)
