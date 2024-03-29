_base_ = [
    "../_base_/models/basic_r50.py",
    "../_base_/datasets/mot16_and_mars.py",
    "../_base_/schedules/basic_schedule.py",
    "../_base_/default_runtime.py",
]
# batch size of 32 and num_frames=16 would use up ~8x7GB of memory
# batch size 32 and num_frames=8 used around 20GB
# batch size 14 and num_frames=8 barely fits on 11GB GPU
num_frames = 4  # FIXME: we should be using 16
model = dict(
    temporal=dict(type="TemporalPooling"),
)
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
)
sampler = dict(
    type="InfiniteNaiveIdentityDistributedSampler",
    batch_size=32,
    num_instances=4,
    shuffle=True,
)
