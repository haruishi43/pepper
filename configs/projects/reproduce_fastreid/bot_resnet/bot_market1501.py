_base_ = [
    "../_base_/models/bot_resnet50.py",
    "../_base_/samplers/default_sampler.py",
    "../_base_/datasets/market1501_REA.py",
    "../_base_/schedules/bot_schedule.py",
    "../_base_/default_runtime.py",
]
# baseline-s + warmup + REA + LS + stride=1 + BNNeck
