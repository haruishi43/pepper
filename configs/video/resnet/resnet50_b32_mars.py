_base_ = [
    "../_base_/models/resnet50.py",
    "../_base_/datasets/mars.py",
    "../_base_/schedules/basic_schedule.py",
    "../_base_/default_runtime.py",
]
sampler = dict(
    num_instances=4,
)
