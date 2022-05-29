# model settings
#norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='vanunet_small',
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=64,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[64, 64, 64, 64],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=256,
        dropout_ratio=0.1,
        num_classes=4,
        norm_cfg=norm_cfg,
        align_corners=False,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]),

    auxiliary_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=0,
        channels=32,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=4,
        norm_cfg=norm_cfg,
        align_corners=False,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))