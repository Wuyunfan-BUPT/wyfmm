# dataset settings
dataset_type = 'BratsDatasetIndiv4C'
data_root = '/kaggle/input/bratsindividual4c'
# img_norm_cfg = dict(
#     mean=[24.371, 41.122, 30.127, 25.215], std=[44.952, 72.598, 56.065, 47.057], to_rgb=True)
img_norm_cfg = dict(
     mean=[0.0, 0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0, 1.0], to_rgb=True)
crop_size = (224, 224)
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged',imdecode_backend='pillow'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged',imdecode_backend='pillow'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(224, 224),
        flip=False,
        transforms=[
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='images/training',
            ann_dir='annotations/training',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test',
        ann_dir='annotations/test',
        pipeline=test_pipeline))
