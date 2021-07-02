import torch
import torch.nn as nn
from models.encoders.base_encoder import BaseEncoder


class ScratchConv(BaseEncoder):

    def __init__(self, opt, kernel_size=3, num_filters=1):

        super(ScratchConv, self).__init__()
        self.proj = nn.Conv2d(1, num_filters, kernel_size, bias=False)

    def forward(self, x):
        x = self.proj(x)
        return x

    def freeze(self):
        for params in self.parameters():
            params.requires_grad = False


class ScratchThreeConv(BaseEncoder):

    def __init__(self, opt, kernel_size=3, num_filters=1):

        super(ScratchThreeConv, self).__init__()
        self.proj1 = nn.Conv2d(1, num_filters, kernel_size, bias=False)
        self.proj2 = nn.Conv2d(1, num_filters, kernel_size, bias=False)
        self.proj3 = nn.Conv2d(1, num_filters, kernel_size, bias=False)

    def forward(self, x):
        x = self.proj1(x)
        x = self.proj2(x)
        x = self.proj3(x)
        return x
