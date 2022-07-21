img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    # dict(
    #     type="Resize",
    #     size=(256, 128),  # (h, w)
    #     interpolation="bilinear",
    # ),
    # dict(
    #     type="RandomCrop",
    #     size=(256, 128),
    #     padding=(10, 10, 10, 10),
    # ),
    dict(
        type="ResizeOrRandom2DTranslation",
        size=(256, 128),
        prob=0.5,
    ),
    dict(
        type="RandomFlip",
        flip_prob=0.5,
        direction="horizontal",
    ),
    dict(
        type="RandomErasing",
        erase_prob=0.5,
        min_area_ratio=0.02,
        max_area_ratio=0.4,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Collect", keys=["img", "gt_label"]),
    dict(type="FormatBundle"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="Resize",
        size=(256, 128),  # (h, w)
        interpolation="bilinear",
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]
data_type = "Market1501Dataset"
data_root = "data/market1501/Market-1501-v15.09.15/"
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    train=dict(
        type=data_type,
        data_prefix=data_root + "bounding_box_train",
        ann_file=data_root + "gtPepper/train.json",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=data_type,
        data_prefix=dict(
            query=data_root + "query",
            gallery=data_root + "bounding_box_test",
        ),
        ann_file=dict(
            query=data_root + "gtPepper/query.json",
            gallery=data_root + "gtPepper/gallery.json",
        ),
        pipeline=test_pipeline,
    ),
    test=dict(
        type=data_type,
        data_prefix=dict(
            query=data_root + "query",
            gallery=data_root + "bounding_box_test",
        ),
        ann_file=dict(
            query=data_root + "gtPepper/query.json",
            gallery=data_root + "gtPepper/gallery.json",
        ),
        pipeline=test_pipeline,
    ),
)
