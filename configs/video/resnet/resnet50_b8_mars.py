_base_ = [
    "../_base_/models/resnet50.py",
    "../_base_/datasets/mars.py",
    "../_base_/schedules/basic_schedule.py",
    "../_base_/default_runtime.py",
]
# batch size of 32 and num_frames=16 would use up ~8x7GB of memory
num_frames = 8  # FIXME: we should be using 16
model = dict(
    temporal=dict(type="TemporalPooling"),
    # temporal=dict(type="RNN", in_channels=2048),
    # temporal=dict(type="TemporalAttention", in_channels=2048, seq_len=num_frames),
)
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
)
sampler = dict(
    batch_size=32,
    num_instances=4,
)
