# model settings
width_mult = 1.0
model = dict(
    type='SingleStageDetector',
    pretrained=True,
    backbone=dict(
        type='mobilenetv2_w1',
        out_indices=(4, 5),
        frozen_stages=-1,
        norm_eval=False),
        # pretrained=True
    # ),
    neck=None,
    bbox_head=dict(
        type='SSDHead',
        num_classes=1,
        in_channels=(int(width_mult * 96), int(width_mult * 320)),
        anchor_generator=dict(
            type='SSDAnchorGeneratorClustered',
            strides=(16, 32),
            widths=(
                [8.0213, 21.4187, 12.544, 29.6107],
                [122.0267, 66.048, 109.9093, 43.6053, 64.512]),
            heights=(
                [12.8, 33.792, 21.76, 53.9307],
                [194.1333, 139.008, 106.24, 89.6853, 61.952]),
            ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=(.0, .0, .0, .0),
            target_stds=(0.1, 0.1, 0.2, 0.2),),
        depthwise_heads=True,
        depthwise_heads_activations='relu',
        loss_balancing=True),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.4,
            neg_iou_thr=0.4,
            min_pos_iou=0.,
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
# optimizer
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1200,
    warmup_ratio=1.0 / 3,
    step=[40000, 55000, 65000])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
dist_params = dict(backend='nccl')
runner = dict(type='IterBasedRunner', max_iters=10000)
log_level = 'INFO'
work_dir = 'outputs/face-detection-0200'
load_from = 'https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/face-detection-0200.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True