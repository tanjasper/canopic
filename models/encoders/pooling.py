"""Pooling Encoders

Examples: max pooling
"""

import torch
import torch.nn as nn
import numpy as np
from models.encoders.util import scale_max_to_1, quantize, normalize
from models.encoders.base_encoder import BaseEncoder
from torch.nn import functional as F


class MaxPool(BaseEncoder):

    def __init__(self, opt, kernel_size, stride=None, padding=0, ceil_mode=False, **kwargs):
        super(MaxPool, self).__init__(**kwargs)
        # in jsons, bools are saved as ints. convert back to bool
        if ceil_mode:
            ceil_mode = True
        else:
            ceil_mode = False
        if stride is None:
            stride = kernel_size
        self.maxpool = nn.MaxPool2d(int(kernel_size), stride=int(stride), padding=padding, ceil_mode=ceil_mode)

    def forward(self, x):
        return self.maxpool(x)


class AveragePool(BaseEncoder):

    def __init__(self, opt, kernel_size, stride=None, padding=0, ceil_mode=False, **kwargs):
        super(AveragePool, self).__init__(**kwargs)
        # in jsons, bools are saved as ints. convert back to bool
        if ceil_mode:
            ceil_mode = True
        else:
            ceil_mode = False
        if stride is None:
            stride = kernel_size
        self.avgpool = nn.AvgPool2d(int(kernel_size), stride=int(stride), padding=padding, ceil_mode=ceil_mode)

    def forward(self, x):
        return self.avgpool(x)


class Upsample(BaseEncoder):

    def __init__(self, opt, scale_factor=None, mode=None, enabled=True, **kwargs):
        super(Upsample, self).__init__(**kwargs)
        self.scale_factor = scale_factor
        self.mode = mode
        self.enabled = enabled

    def forward(self, x):
        if self.enabled:
            return F.interpolate(x, scale_factor=int(self.scale_factor), mode=self.mode)
        else:
            return x
