_base_ = [
    './coco_data_pipeline.py'
]
# model settings
input_size = 512
image_width, image_height = input_size, input_size
width_mult = 1.0
model = dict(
    type='SingleStageDetector',
    pretrained=True,
    backbone=dict(
        type='mobilenetv2_w1',
        out_indices=(4, 5),
        frozen_stages=-1,
        norm_eval=False
    ),
    neck=None,
    bbox_head=dict(
        type='SSDHead',
        num_classes=3,
        in_channels=(int(width_mult * 96), int(width_mult * 320)),
        anchor_generator=dict(
            type='SSDAnchorGeneratorClustered',
            strides=(16, 32),
            widths=[
                [image_width * x for x in
                 [0.027985084698552577, 0.05080701904477674, 0.05982521591620597,
                  0.09207979657583232]],
                [image_width * x for x in
                 [0.20259559591239545, 0.12732159067712165, 0.1934764607279801,
                  0.35789819765821385, 0.587237190090815]],
            ],
            heights=[
                [image_height * x for x in
                 [0.0479932750007942, 0.11405624363135988, 0.20898520700595447,
                  0.3335612473329943]],
                [image_height * x for x in
                 [0.1674597285977332, 0.4942406505865888, 0.7431758023015786, 0.41062433524702613,
                  0.5719883460663506]],
            ],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=(0.0, 0.0, 0.0, 0.0),
            target_stds=(0.1, 0.1, 0.2, 0.2), ),
        depthwise_heads=True,
        depthwise_heads_activations='relu',
        loss_balancing=True),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.4,
            neg_iou_thr=0.4,
            min_pos_iou=0.0,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        smoothl1_beta=1.,
        use_giou=False,
        use_focal=False,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200))
evaluation = dict(interval=1000, metric='mAP')
# optimizer
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1200,
    warmup_ratio=1.0 / 3,
    step=[8000, 15000, 18000])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
dist_params = dict(backend='nccl')
runner = dict(type='IterBasedRunner', max_iters=13000)
log_level = 'INFO'
work_dir = 'outputs/person-vehicle-bike-detection-2002'
load_from = 'https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/vehicle-person-bike-detection-2002-1.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
