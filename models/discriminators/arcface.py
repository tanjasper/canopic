"""Obtained from https://github.com/ronghuaiyang/arcface-pytorch"""

from models.discriminators.base_discriminator import BaseDiscriminator
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from . import Googlenet_64, resnet18_64_N_IN


class ArcfaceDiscriminator(BaseDiscriminator):
    """Discriminator with arcface metric at the end"""

    def __init__(self, base_arch=None, base_settings=None, embedding_dim=512, num_classes=None, easy_margin=False, s=64, m=0.5):

        super(ArcfaceDiscriminator, self).__init__()
        base_net_class = globals()[base_arch]
        self.base = base_net_class(**base_settings)
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):

        x = self.base(x)
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output
