img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
num_frames = 4  # FIXME: we should use 16
train_pipeline = [
    dict(type="VideoSampler", method="random_crop", seq_len=num_frames),
    dict(type="LoadMultiImagesFromFile", to_float32=True),
    # dict(
    #     type="SeqProbRandomResizedCrop",
    #     size=(256, 128),
    #     scale=(0.888, 1.0),
    #     crop_prob=0.5,
    # ),
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
        size=(256, 128),  # (h, w)
        interpolation="bilinear",
    ),
    dict(type="SeqNormalize", **img_norm_cfg),
    dict(type="VideoCollect", keys=["img"]),
    dict(type="FormatVideoEval", as_list=False),
]
data_type = "VideoDataset"
data_root = "data/mars/"
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=6,
    train=dict(
        type=data_type,
        data_prefix=data_root,
        ann_file=data_root + "gtPepper/train.json",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=data_type,
        data_prefix=dict(
            query=data_root,
            gallery=data_root,
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
            query=data_root,
            gallery=data_root,
        ),
        ann_file=dict(
            query=data_root + "gtPepper/query.json",
            gallery=data_root + "gtPepper/gallery.json",
        ),
        pipeline=test_pipeline,
    ),
)
