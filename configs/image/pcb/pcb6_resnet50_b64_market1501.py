_base_ = [
    "../_base_/models/pcb_resnet50.py",
    "../_base_/samplers/infinite_balanced.py",
    "../_base_/datasets/market1501.py",
    "../_base_/schedules/bot_schedule.py",
    "../_base_/default_runtime.py",
]
num_parts = 6
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
)
sampler = dict(
    batch_size=64,
    num_instances=4,
)
model = dict(
    neck=dict(type="PartPooling", num_parts=num_parts),
    head=dict(
        num_parts=num_parts,
        loss_cls=dict(
            type="LabelSmoothLoss", label_smooth_val=0.1, loss_weight=1.0 / num_parts
        ),
    )
)
