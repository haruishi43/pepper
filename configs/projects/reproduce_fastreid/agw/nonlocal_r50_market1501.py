_base_ = [
    "../_base_/models/basic_resnet50.py",
    "../_base_/datasets/market1501_REA.py",
    "../_base_/schedules/bot_schedule.py",
    "../_base_/default_runtime.py",
]
model = dict(
    backbone=dict(
        type="BetterPlugResNet",
        depth=50,
        num_stages=4,
        strides=(1, 2, 2, 1),  # increases final features by x2
        out_indices=(3,),
        style="pytorch",
        plugins=[
            dict(  # 2nd
                cfg=dict(type='NonLocal2d', mode="dot_product"),
                stages=(False, True, False, False),
                layers=(2, 3),
                position='after_conv3',
            ),
            dict(  # 3rd
                cfg=dict(type='NonLocal2d', mode="dot_product"),
                stages=(False, False, True, False),
                layers=(3, 4, 5),
                position='after_conv3',
            ),
        ],
    ),
    neck=dict(type="GeneralizedMeanPooling"),
)
