_base_ = [
    "../_base_/models/basic_resnet50.py",
    "../_base_/datasets/market1501.py",
    "../_base_/schedules/default_schedule.py",
    "../_base_/default_runtime.py",
]
# baseline-s
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
)
sampler = dict(
    batch_size=64,
    num_instances=4,
)
