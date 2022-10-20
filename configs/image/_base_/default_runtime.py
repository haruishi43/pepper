# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook'),
    ],
)
# yapf:enable
# custom_hooks = [dict(type="NumClassCheckHook")]
# custom_hooks = [  # debug
#     dict(type="NumClassCheckHook"),
#     dict(type="CheckInvalidLossHook", interval=50),
# ]

dist_params = dict(backend="nccl")
log_level = "INFO"  # NOTE: change this to `DEBUG`
load_from = None
resume_from = None
workflow = [("train", 1)]

# This is added to other mm-projects
# cudnn_benchmark = True
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = "fork"

# Unused parameters: https://github.com/open-mmlab/mmcv/issues/1601
find_unused_parameters = True
