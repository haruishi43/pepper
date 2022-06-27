_base_ = [
    "../_base_/models/resnet50.py",
    "../_base_/datasets/image/market1501.py",
    "../_base_/schedules/schedule_8k.py",
    "../_base_/default_runtime.py",
]
sampler = dict(
    num_instances=4,
)
