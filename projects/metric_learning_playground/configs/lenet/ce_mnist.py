_base_ = [
    "../_base_/models/lenetplusplus.py",
    "../_base_/datasets/mnist_orig.py",
    "../_base_/schedules/default_schedule.py",
    "../_base_/default_runtime.py",
]

model = dict(
    head=dict(
        type="VisualizeFeatureHead",
        vis_dim=2,
        loss_pairwise=None,  # only ce
    )
)

optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
)

work_dir = './work_dirs/ce_mnist/'
