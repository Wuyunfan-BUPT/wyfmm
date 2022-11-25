# dataset settings
dataset_type = 'BratsDataset2020'
data_root = '/kaggle/input/mydataset2'
crop_size = (224, 224)
#crop_size = (192, 192)
classes = ('BG', 'WT', 'TC', 'ET')
palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34]]
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged',imdecode_backend='tifffile'),
    dict(type='LoadAnnotations'),
    # dict(type='RandomMosaic', prob=1),
    dict(type='Resize', img_scale=crop_size, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

# train_dataset = dict(
#     # type='MultiImageMixDataset',
#     dataset=dict(
#         classes=classes,
#         palette=palette,
#         type=dataset_type,
#         reduce_zero_label=False,
#         img_dir=data_root + "images/train",
#         ann_dir=data_root + "annotations/train",
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(type='LoadAnnotations'),
#         ]
#     ),
#     pipeline=train_pipeline
# )

validation_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged',imdecode_backend='tifffile'),
    dict(
        type='MultiScaleFlipAug', # 测试时数据增强
        img_scale=crop_size,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]
    )
    #dict(type='LoadAnnotations'),
    #dict(type='Resize', img_scale=(224,224)),
    #dict(type='Resize', img_scale=(224,224), ratio_range=(0.5, 2.0)),
    #dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='RandomFlip', prob=0.5),

    # dict(type='DefaultFormatBundle'),
    #dict(type='ImageToTensor', keys=['img']),
    # dict(type='Collect', keys=['img'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged',imdecode_backend='tifffile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
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
        pipeline=validation_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test',
        ann_dir='annotations/test',
        pipeline=test_pipeline))
