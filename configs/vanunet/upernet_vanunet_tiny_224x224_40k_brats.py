_base_ = [
    '../_base_/models/upernet_vanUnet1.py', '../_base_/datasets/brats.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
model = dict(
    pretrained='checkpoints/van_tiny_seg.pth',
    backbone=dict(
        #init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        img_size=224,
        in_chans=3,
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        dec_num_convs=(2, 2, 2, 2),
        upsample_cfg=dict(type='InterpConv')
        ),
    decode_head=dict(in_channels=[32, 64, 160, 256], num_classes=4),
    auxiliary_head=dict(in_channels=160, num_classes=4))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'head': dict(lr_mult=10.),
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
