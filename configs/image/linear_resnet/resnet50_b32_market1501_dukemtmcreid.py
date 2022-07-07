_base_ = [
    "../_base_/models/linear_resnet50.py",
    "../_base_/datasets/market1501.py",
    "../_base_/schedules/basic_schedule.py",
    "../_base_/default_runtime.py",
]
sampler = dict(
    num_instances=4,
)
