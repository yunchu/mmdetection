_base_ = [
    './coco_data_pipeline.py'
]
# model settings
width_mult = 1.0
model = dict(
    type='ATSS',
    pretrained=True,
    backbone=dict(
        type='resnet152b',
        out_indices=(1, 2, 3, 4),
        frozen_stages=-1,
        norm_eval=False),
    neck=dict(
        type='RSSH_FPN',
        in_channels=[int(width_mult * 256),
                     int(width_mult * 512),
                     int(width_mult * 1024),
                     int(width_mult * 2048),
                     ],
        out_channels=256,
        start_level=0,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='ATSSHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=128,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.5),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=750))
evaluation = dict(interval=1000, metric='mAP')
# optimizer
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
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
work_dir = 'outputs/face-detection-0206'
load_from = 'https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/face-detection-0206.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True