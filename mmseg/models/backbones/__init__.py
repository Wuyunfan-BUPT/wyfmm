# Copyright (c) OpenMMLab. All rights reserved.
from .beit import BEiT
from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2
from .cgnet import CGNet
from .erfnet import ERFNet
from .fast_scnn import FastSCNN
from .hrnet import HRNet
from .icnet import ICNet
from .mae import MAE
from .mit import MixVisionTransformer
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1c, ResNetV1d
from .resnext import ResNeXt
from .stdc import STDCContextPathNet, STDCNet
from .swin import SwinTransformer
from .timm_backbone import TIMMBackbone
from .twins import PCPVT, SVT
from .unet import UNet
from .vit import VisionTransformer
from .swinL_convR_unet import SwinConvUnet
from .conv_swin_Unet import convSwinUnet
from .vanUnet import VANUnet
from .unet_reverse import UNet_re
from .van import VAN
from .swinL_convR_unet1 import SwinConvUnet1
from .swinL_convR_unet2 import SwinConvUnet2
from .swin_conv1 import SwinTransformer1
from .swin_conv2 import SwinTransformer2
from .swinL_convR_unet0 import SwinConvUnet0
from .swinP import SwinTransformerP
from .swinPure import SwinTransformerPure
from .swinPureConv import SwinTransformerPureUnet
from .swinPureConv1 import SwinTransformerPureUnet1
from .swinPureConv2 import SwinTransformerPureUnet2

__all__ = [
    'ResNet', 'ResNetV1c', 'ResNetV1d', 'ResNeXt', 'HRNet', 'FastSCNN',
    'ResNeSt', 'MobileNetV2', 'UNet', 'CGNet', 'MobileNetV3',
    'VisionTransformer', 'SwinTransformer', 'MixVisionTransformer',
    'BiSeNetV1', 'BiSeNetV2', 'ICNet', 'TIMMBackbone', 'ERFNet', 'PCPVT',
    'SVT', 'STDCNet', 'STDCContextPathNet', 'BEiT', 'MAE', 'SwinConvUnet',
    'convSwinUnet', 'VANUnet', 'UNet_re', 'VAN', 'SwinConvUnet1', 'SwinConvUnet2',
    'SwinTransformer1','SwinTransformer2', 'SwinConvUnet0', 'SwinTransformerP',
    'SwinTransformerPure', 'SwinTransformerPureUnet', 'SwinTransformerPureUnet1',
    'SwinTransformerPureUnet2'
]
