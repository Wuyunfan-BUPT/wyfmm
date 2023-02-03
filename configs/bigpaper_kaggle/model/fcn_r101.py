_base_ = [
    '../../_base_/models/fcn_r50-d8.py', '../../_base_/datasets/bratsIndividual4C.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_160k_dice.py'
]
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101),
        decode_head=dict(num_classes=4), auxiliary_head=dict(num_classes=4))

