_base_ = [
    "../_base_/models/pcb_resnet50.py",
    "../_base_/samplers/infinite_balanced.py",
    "../_base_/datasets/market1501.py",
    "../_base_/schedules/bot_schedule.py",
    "../_base_/default_runtime.py",
]
num_parts = 4
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
)
sampler = dict(
    batch_size=64,
    num_instances=4,
)
model = dict(
    head=dict(
        loss_pairwise=dict(type="TripletLoss", margin=0.3, loss_weight=1.0 / num_parts),
    )
)
