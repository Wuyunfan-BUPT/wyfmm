_base_ = [
    '../../_base_/models/P_P_fpn_swin.py',  '../../_base_/datasets/brats2020_zcrossP_kaggle.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_160k_dice.py'
]
# '../../_base_/datasets/bratsIndividual4C.py',
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=64,
        patch_size=2,
        mlp_ratio=4,
        depths=[2, 2, 4, 8, 2],
        num_heads=[2, 4, 8, 16, 32],
        strides=(2, 2, 2, 2, 2),
        out_indices=(0, 1, 2, 3, 4),
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512, 1024],
        out_channels=64,
        num_outs=5),
    decode_head=dict(
        type='FPNHead',
        in_channels=[64, 64, 64, 64, 64],
        in_index=[0, 1, 2, 3, 4],
        feature_strides=[2, 4, 8, 16, 32],
        channels=64,
        dropout_ratio=0.1,
        num_classes=4,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]),
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
