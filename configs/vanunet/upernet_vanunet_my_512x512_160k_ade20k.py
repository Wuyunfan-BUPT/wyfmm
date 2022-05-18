_base_ = [
    '../_base_/models/upernet_vanUnet1.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
checkpoint_file = None
model = dict(
    pretrained=None, #'checkpoints/van_tiny_seg.pth',
    backbone=dict(
        type='VANUnet',
        img_size=224,
        in_chans=3,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        num_stages=4,
        linear=False,
        dec_num_convs=(2, 2, 2, 2),
        dec_dilations=(1, 1, 1, 1),
        conv_cfg=None,
        upsample_cfg=dict(type='InterpConv'),
        conv_norm_cfg=dict(type='BN'),
        conv_act_cfg=dict(type='ReLU')
    ),
    decode_head=dict(in_channels=[64, 128, 256, 512],num_classes=150),
    auxiliary_head=dict(in_channels=256,num_classes=150))

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
