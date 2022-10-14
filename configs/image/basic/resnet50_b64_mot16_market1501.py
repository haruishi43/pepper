_base_ = [
    "../_base_/models/basic_resnet50.py",
    "../_base_/datasets/mot16_and_market1501.py",
    "../_base_/schedules/bot_schedule.py",
    "../_base_/default_runtime.py",
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
)

# NOTE: for MOT, we don't have camids so we must use the naive sampler
sampler = dict(
    type="InfiniteNaiveIdentityDistributedSampler",
    batch_size=64,
    num_instances=4,
    shuffle=True,
)
