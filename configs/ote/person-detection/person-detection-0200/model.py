_base_ = [
    './coco_data_pipeline.py'
]
# model settings
input_size = 256
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
        num_classes=1,
        in_channels=(int(width_mult * 96), int(width_mult * 320)),
        anchor_generator=dict(
            type='SSDAnchorGeneratorClustered',
            strides=(16, 32),
            widths=[
                [image_width * x for x in
                 [0.015411783166343854, 0.033018232306549156, 0.04467156688464953,
                  0.0610697815328886]],
                [image_width * x for x in
                 [0.0789599954420517, 0.10113984043326349, 0.12805187473050397, 0.16198319380154758,
                  0.21636496806213493]],
            ],
            heights=[
                [image_height * x for x in
                 [0.05032631418898226, 0.10070800135152037, 0.15806180366055939,
                  0.22343401646383804]],
                [image_height * x for x in
                 [0.300881401352503, 0.393181898580379, 0.4998807213337051, 0.6386035764261081,
                  0.8363451552091518]],
            ],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=(0.0, 0.0, 0.0, 0.0),
            target_stds=(0.1, 0.1, 0.2, 0.2), ),
        depthwise_heads=True,
        depthwise_heads_activations='relu',
        loss_balancing=True),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.4,
            neg_iou_thr=0.4,
            min_pos_iou=0.0,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        smoothl1_beta=1.0,
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
work_dir = 'outputs/person-detection-0200'
load_from = 'https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/person-detection-0200-1.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
