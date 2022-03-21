img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadMultiImagesFromFile", to_float32=True),
    dict(
        type="SeqResize",
        size=(256, 128),  # (h, w)
        interpolation="bilinear",
    ),
    dict(
        type="SeqRandomFlip",
        flip_prob=0.5,
        direction="horizontal",
    ),
    dict(type="SeqNormalize", **img_norm_cfg),
    dict(type='VideoCollect', keys=['img', 'gt_label']),
    dict(type='FormatBundle')
]
test_pipeline = [
    dict(type="LoadMultiImageFromFile"),
    dict(
        type="SeqResize",
        size=(256, 128),  # (h, w)
        interpolation="bilinear",
    ),
    dict(type="SeqNormalize", **img_norm_cfg),
    dict(type="VideoCollect", keys=["img"], meta_keys=[]),
]
data_type = "BaseDataset"
data_root = "tests/data/mini_mars/"
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=data_type,
        data_prefix=data_root + "bbox_train",
        ann_file=data_root + "gtPepper/train.json",
        pipeline=train_pipeline,
    ),
    query=dict(
        type=data_type,
        data_prefix=data_root + "bbox_test",
        ann_file=data_root + "gtPepper/query.json",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=data_type,
        data_prefix=data_root + "bbox_test",
        ann_file=data_root + "gtPepper/gallery.json",
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1, metric="mAP")
