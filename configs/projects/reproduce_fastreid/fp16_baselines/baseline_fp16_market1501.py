_base_ = [
    "../_base_/models/basic_resnet50.py",
    "../_base_/samplers/default_sampler.py",
    "../_base_/datasets/market1501.py",
    "../_base_/schedules/default_fp16_schedule.py",
    "../_base_/default_runtime.py",
]
