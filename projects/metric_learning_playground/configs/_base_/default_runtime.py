# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ]
)
# yapf:enable

evaluation = dict(
    interval=500, metric='accuracy', metric_options={'topk': (1, )}
)

# checkpoint saving
checkpoint_config = dict(interval=500)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=6000)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
