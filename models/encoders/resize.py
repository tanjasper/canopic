import torch
from models.encoders.base_encoder import BaseEncoder
from torch.nn import functional as F

class FasterRCNNResize(BaseEncoder):

    def __init__(self, opt, min_dim_size=400, **kwargs):
        super(FasterRCNNResize, self).__init__(**kwargs)
        self.min_dim_size = min_dim_size

    def forward(self, x):
        if min(x.shape[2], x.shape[3]) < self.min_dim_size:
            return F.interpolate(x, size=self.min_dim_size)
        else:
            return x
