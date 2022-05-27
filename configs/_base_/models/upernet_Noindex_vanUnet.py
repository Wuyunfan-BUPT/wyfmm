# model settings
#norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='vanunet_tiny',
        style='pytorch'),
    decode_head=dict(
        type='UPerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        #sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        ignore_index=0,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]),
        #loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, avg_non_ignore=True)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=160,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        #sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        ignore_index=0,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]),
        #loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,  loss_weight=1.0, avg_non_ignore=True)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))