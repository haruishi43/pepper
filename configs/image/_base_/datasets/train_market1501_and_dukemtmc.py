img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
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

# train on market and duke
# evaluate on market and duke
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=6,
    train=[
        dict(
            type="Market1501Dataset",
            data_prefix="data/market1501/Market-1501-v15.09.15/"
            + "bounding_box_train",
            ann_file="data/market1501/Market-1501-v15.09.15/"
            + "gtPepper/train.json",
            pipeline=train_pipeline,
        ),
        dict(
            type="ImageDataset",
            data_prefix="data/dukemtmc-reid/DukeMTMC-reID/"
            + "bounding_box_train",
            ann_file="data/dukemtmc-reid/DukeMTMC-reID/"
            + "gtPepper/train.json",
            pipeline=train_pipeline,
        ),
    ],
    val=[
        dict(
            type="Market1501Dataset",
            data_prefix=dict(
                query="data/market1501/Market-1501-v15.09.15/" + "query",
                gallery="data/market1501/Market-1501-v15.09.15/"
                + "bounding_box_test",
            ),
            ann_file=dict(
                query="data/market1501/Market-1501-v15.09.15/"
                + "gtPepper/query.json",
                gallery="data/market1501/Market-1501-v15.09.15/"
                + "gtPepper/gallery.json",
            ),
            pipeline=test_pipeline,
        ),
        dict(
            type="ImageDataset",
            data_prefix=dict(
                query="data/dukemtmc-reid/DukeMTMC-reID/" + "query",
                gallery="data/dukemtmc-reid/DukeMTMC-reID/"
                + "bounding_box_test",
            ),
            ann_file=dict(
                query="data/dukemtmc-reid/DukeMTMC-reID/"
                + "gtPepper/query.json",
                gallery="data/dukemtmc-reid/DukeMTMC-reID/"
                + "gtPepper/gallery.json",
            ),
            pipeline=test_pipeline,
        ),
    ],
    test=[
        dict(
            type="Market1501Dataset",
            data_prefix=dict(
                query="data/market1501/Market-1501-v15.09.15/" + "query",
                gallery="data/market1501/Market-1501-v15.09.15/"
                + "bounding_box_test",
            ),
            ann_file=dict(
                query="data/market1501/Market-1501-v15.09.15/"
                + "gtPepper/query.json",
                gallery="data/market1501/Market-1501-v15.09.15/"
                + "gtPepper/gallery.json",
            ),
            pipeline=test_pipeline,
        ),
        dict(
            type="ImageDataset",
            data_prefix=dict(
                query="data/dukemtmc-reid/DukeMTMC-reID/" + "query",
                gallery="data/dukemtmc-reid/DukeMTMC-reID/"
                + "bounding_box_test",
            ),
            ann_file=dict(
                query="data/dukemtmc-reid/DukeMTMC-reID/"
                + "gtPepper/query.json",
                gallery="data/dukemtmc-reid/DukeMTMC-reID/"
                + "gtPepper/gallery.json",
            ),
            pipeline=test_pipeline,
        ),
    ],
)
