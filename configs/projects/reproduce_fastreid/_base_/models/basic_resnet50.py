# DEBUG: BaseineHead is not working correctly!

model = dict(
    type="ImageReID",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(3,),
        style="pytorch",
        # init_cfg=dict(
        #     type="Pretrained",
        #     checkpoint="torchvision://resnet50",
        # ),
        init_cfg=dict(
            type="Pretrained",
            checkpoint="mmcls://resnet50",
        ),
    ),
    neck=dict(type="GlobalAveragePooling", dim=2),
    head=dict(
        # type="BasicHead",
        type="BaselineHead",
        in_channels=2048,
        num_classes=380,
        loss_cls=dict(type="CrossEntropyLoss", loss_weight=1.0),
        loss_pairwise=dict(type="TripletLoss", margin=0.3, loss_weight=1.0),
        norm_cfg=dict(type="BN1d"),
        act_cfg=dict(type="ReLU"),
        init_cfg=dict(type="Normal", layer="Linear", mean=0, std=0.001, bias=0),
        # init_cfg=[
        #     dict(type="Normal", layer="Linear", mean=0, std=0.001, bias=0),
        #     dict(type="Constant", layer="BatchNorm", val=1.0, bias=0.0),
        # ],
    ),
    # init_cfg=dict(
    #     type="Pretrained",
    #     checkpoint="https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth",  # noqa: E251  # noqa: E501
    # ),
    # init_cfg=dict(
    #     type="Pretrained",
    #     checkpoint="https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth",  # noqa: E251  # noqa: E501
    # ),
    # init_cfg=dict(
    #     type="Pretrained",
    #     checkpoint="https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_3rdparty-mill_in21k_20220331-faac000b.pth",  # noqa: E251  # noqa: E501
    # ),
    inference_stage="pre_logits",
)
