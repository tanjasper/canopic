import torch
import os
import torch.nn as nn
import numpy as np
from models.encoders.base_encoder import BaseEncoder


class FixedQuantize(BaseEncoder):

    def __init__(self, opt, bits, **kwargs):
        super(FixedQuantize, self).__init__(**kwargs)
        self.bits = bits

    def forward(self, x):
        qmin = 0.
        qmax = 2. ** self.bits - 1.
        min_value = x.view(x.size(0), -1).min(1)[0].view(-1, 1, 1, 1)  # min values for each input
        max_value = x.view(x.size(0), -1).max(1)[0].view(-1, 1, 1, 1)  # max values for each input
        scale_value = (max_value - min_value) / (qmax - qmin)
        scale_value = torch.clamp(scale_value, 1e-4)
        x = ((x - min_value) / ((max_value - min_value) + 1e-4)) * (qmax - qmin)
        # vivek.hack.added
        x = x.clamp(qmin, qmax)
        y = x.detach() # detach to round and prop gradient through round
        y = y - y.round()
        x = x - y
        x = x.add(-qmin).mul(scale_value).add(min_value)  # dequantize
        return x

def heaviside(x):
    a = x>=0
    a = a.float()
    return a

class LearnQuantize(BaseEncoder):
    """Encoder for learned quantization

    Settings:
        bits (int): number of bits for quantization
        sigma_scale
    """
    def __init__(self, opt, bits, sigma_scale=5, exact_quantize=False, **kwargs):
        super(LearnQuantize, self).__init__(**kwargs)
        self.exact_quantize = exact_quantize
        self.sigma_scale = sigma_scale
        self.bits = bits
        self.bias = nn.Parameter(
            torch.from_numpy(np.linspace(0, 2 ** self.bits - 1, 2 ** self.bits, dtype='float32')[:-1] + 0.5))
        self.levels = np.linspace(0, 2 ** self.bits - 1, 2 ** self.bits, dtype='float32')[:-1]

    def forward(self, x):
        if self.exact_quantize:
            qmin = 0.
            qmax = 2. ** self.bits - 1.
            min_value = x.view(x.size(0), -1).min(1)[0].view(-1, 1, 1, 1)  # min values for each input
            max_value = x.view(x.size(0), -1).max(1)[0].view(-1, 1, 1, 1)  # max values for each input
            scale_value = (max_value - min_value) / (qmax - qmin)
            scale_value = torch.clamp(scale_value, 1e-4)
            x = ((x - min_value) / ((max_value - min_value) + 1e-4)) * (qmax - qmin)
            y = torch.zeros(x.shape, device=x.device)
            self.bias.data = self.bias.data.clamp(0, (2 ** self.bits - 1))
            self.bias.data = self.bias.data.sort(0)[0]
            for i in range(self.levels.shape[0]):
                y = y + heaviside(x - self.bias[i])
            y = y.mul(scale_value).add(min_value)
            return y
        else:
            qmin = 0.
            qmax = 2. ** self.bits - 1.
            min_value = x.view(x.size(0), -1).min(1)[0].view(-1, 1, 1, 1)  # min values for each input
            max_value = x.view(x.size(0), -1).max(1)[0].view(-1, 1, 1, 1)  # max values for each input
            scale_value = (max_value - min_value) / (qmax - qmin)
            scale_value = torch.clamp(scale_value, 1e-4)
            x = ((x - min_value) / ((max_value - min_value) + 1e-4)) * (qmax - qmin)
            y = torch.zeros(x.shape, device=x.device)
            self.bias.data = self.bias.data.clamp(0, (2 ** self.bits - 1))
            self.bias.data = self.bias.data.sort(0)[0]
            for i in range(self.levels.shape[0]):
                y = y + torch.sigmoid(self.sigma_scale * (x - self.bias[i]))
            y = y.mul(scale_value).add(min_value)
            return y

    def save_debug_data(self, save_dir, savename_suffix=''):
        savename = 'quantize_bias_' + savename_suffix
        np.save(os.path.join(save_dir, savename), self.bias.detach().cpu().numpy())

    def train_generator_update(self, epoch):
        self.sigma_scale = 5*(epoch+1)
        pass
