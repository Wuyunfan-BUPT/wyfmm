#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Project : Guangfa UI test
# @Time    : 2022/11/2 21:42
# @Author  : wuyfee
# @File    : facal_wyf_loss.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from .utils import weight_reduce_loss


@LOSSES.register_module()
class FocalLoss_wyf(nn.Module):
    def __init__(self, gamma=2, alpha=1, size_average=True, loss_name='loss_focal_wyf'):
        super(FocalLoss_wyf, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
        self._loss_name = loss_name

    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        assert (logits.size(0) == labels.size(0))
        assert (logits.size(2) == labels.size(1))
        batch_size = logits.size(0)
        labels_length = logits.size(1)
        seq_length = logits.size(2)

        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([batch_size, labels_length, seq_length]).scatter_(1, new_label, 1)

        # calculate log
        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt) ** self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name