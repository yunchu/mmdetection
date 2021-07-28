_base_ = [
    './coco_data_pipeline.py'
]
# model settings
width_mult = 1.0
model = dict(
    type='FCOS',
    pretrained=True,
    backbone=dict(
        type='mobilenetv2_w1',
        out_indices=(3, 4, 5),
        frozen_stages=-1,
        norm_eval=False),
    neck=dict(
        type='FPN',
        in_channels=[int(width_mult * 32), int(width_mult * 96), int(width_mult * 320)],
        out_channels=48,
        start_level=0,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=1,
        in_channels=48,
        stacked_convs=4,
        feat_channels=32,
        strides=(8, 16, 32, 64, 128),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
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
    step=[40000, 55000, 65000])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
dist_params = dict(backend='nccl')
runner = dict(type='IterBasedRunner', max_iters=10000)
log_level = 'INFO'
work_dir = 'outputs/face-detection-0205'
load_from = 'https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/object_detection/v2/face-detection-0205-retrained.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True