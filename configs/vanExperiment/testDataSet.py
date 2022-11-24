_base_ = [
    '../_base_/models/upernet_vanUnet.py', '../_base_/datasets/brats2020.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k_dice.py'
]
model = dict(
    pretrained='./work_dirs/van/latest.pth',
    backbone=dict(
        type='vanunet_small',
        style='pytorch'),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=4,
        # sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_name='loss_ce', class_weight=[0.1,1.0,1.0,1.0], loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', ignore_index=0, loss_weight=3.0, avg_non_ignore=True)]
    ),
    auxiliary_head=dict(
        in_channels=320,
        num_classes=4,
        # sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_name='loss_ce', class_weight=[0.1,1.0,1.0,1.0], loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', ignore_index=0, loss_weight=3.0, avg_non_ignore=True)]

    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2)
