img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
num_frames = 4  # FIXME: we should use 16
frame_size = (224, 112)
train_pipeline = [
    dict(type="VideoSampler", method="random_crop", seq_len=num_frames),
    dict(type="LoadMultiImagesFromFile", to_float32=True),
    dict(
        type="SeqResizeOrRandom2DTranslation",
        size=frame_size,
        prob=0.5,
    ),
    dict(
        type="SeqRandomFlip",
        flip_prob=0.5,
        direction="horizontal",
    ),
    dict(
        type="SeqRandomErasing",
        share_params=False,
        erase_prob=0.5,
        min_area_ratio=0.02,
        max_area_ratio=0.4,
    ),
    dict(type="SeqNormalize", **img_norm_cfg),
    dict(type="VideoCollect", keys=["img", "gt_label"]),
    dict(type="FormatBundle"),
]
test_pipeline = [
    dict(type="VideoSampler", method="evenly", seq_len=num_frames),
    dict(type="LoadMultiImagesFromFile"),
    dict(
        type="SeqResize",
        size=frame_size,
        interpolation="bilinear",
    ),
    dict(type="SeqNormalize", **img_norm_cfg),
    dict(type="VideoCollect", keys=["img"]),
    dict(type="FormatVideoEval", as_list=False),
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=6,
    train=[
        dict(
            type="VideoDataset",
            data_prefix="data/mars/",
            ann_file="data/mars/" + "gtPepper/train.json",
            pipeline=train_pipeline,
        ),
        dict(
            type="MOT16VideoDataset",
            data_prefix=None,
            ann_file="data/MOT16/gtPepper/train_vid.json",
            pipeline=train_pipeline,
            train_seq=("02", "04", "05", "09", "10"),
            vis_ratio=0.7,
            vis_frame_ratio=0.6,
            min_seq_len=8,
        ),
    ],
    val=dict(
        type="VideoDataset",
        data_prefix=dict(
            query="data/mars/",
            gallery="data/mars/",
        ),
        ann_file=dict(
            query="data/mars/" + "gtPepper/query.json",
            gallery="data/mars/" + "gtPepper/gallery.json",
        ),
        pipeline=test_pipeline,
    ),
    test=dict(
        type="VideoDataset",
        data_prefix=dict(
            query="data/mars/",
            gallery="data/mars/",
        ),
        ann_file=dict(
            query="data/mars/" + "gtPepper/query.json",
            gallery="data/mars/" + "gtPepper/gallery.json",
        ),
        pipeline=test_pipeline,
    ),
)
