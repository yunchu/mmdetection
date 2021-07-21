# dataset settings
dataset_type = 'OTEDataset'
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromOTEDataset', to_float32=True),
    dict(type='LoadAnnotationFromOTEDataset', with_bbox=True, with_label=True),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(
        type='Resize',
        img_scale=[(704, 704), (844, 704), (704, 844), (564, 704), (704, 564)],
        multiscale_mode='value',
        keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromOTEDataset'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(704, 704),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ote_dataset=None,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ote_dataset=None,
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ote_dataset=None,
        test_mode=True,
        pipeline=test_pipeline))
