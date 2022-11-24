# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class BratsDataset2020(CustomDataset):
    """STARE dataset.

    In segmentation map annotation for STARE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.ah.png'.
    """

    CLASSES = ('BG', 'WT', 'TC', 'ET')

    PALETTE = [[255, 255, 255], [129, 127, 38], [120, 69, 125], [53, 125, 34]]

    def __init__(self, **kwargs):
        super(BratsDataset2020, self).__init__(
            img_suffix='.tiff',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
