_base_ = [
    '../../../_base_/models/upernet_vanUnet.py', '../../../_base_/datasets/bratsCombine4C.py',
    '../../../_base_/default_runtime.py', '../../../_base_/schedules/schedule_40k_dice.py'
]
model = dict(
    pretrained='./work_dirs/van/latest.pth',
    backbone=dict(
        type='vanunet_small',
        style='pytorch'),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=4
    ),
    auxiliary_head=dict(
        in_channels=320,
        num_classes=4
    ))
# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2)
