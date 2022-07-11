_base_ = [
    "../../video/_base_/models/basic_resnet50.py",
    "../../video/_base_/datasets/mars.py",
    "../../video/_base_/schedules/basic_schedule.py",
    "../../video/_base_/default_runtime.py",
]
# batch size of 32 and num_frames=16 would use up ~8x7GB of memory
# batch size 32 and num_frames=8 used around 20GB
# batch size 14 and num_frames=8 barely fits on 11GB GPU
num_frames = 4  # FIXME: we should be using 16
model = dict(
    neck=None,
    temporal=dict(
        type="TemporalConvAttention",
        in_channels=2048,
        mid_dim=256,
        attention_mode="softmax",
        kernel_size=(7, 4),
    ),
)
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
)
sampler = dict(
    batch_size=32,
    num_instances=4,
)
