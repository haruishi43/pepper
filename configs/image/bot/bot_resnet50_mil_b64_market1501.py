_base_ = [
    "../_base_/models/bot_resnet50.py",
    "../_base_/samplers/infinite_balanced.py",
    "../_base_/datasets/market1501.py",
    "../_base_/schedules/bot_schedule.py",
    "../_base_/default_runtime.py",
]

# maybe using different pretrained weights help
model = dict(
    init_cfg=dict(
        type="Pretrained",
        checkpoint="https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_3rdparty-mill_in21k_20220331-faac000b.pth",  # noqa: E251  # noqa: E501
    ),
)
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
)
sampler = dict(
    batch_size=64,
    num_instances=4,
)
evaluation = dict(dist_metric="cosine")
