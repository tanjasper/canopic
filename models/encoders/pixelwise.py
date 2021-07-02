"""Pixelwise Encoders

Included here: gamma
"""

import torch
import torch.nn as nn
from models.encoders.util import scale_max_to_1, quantize, normalize
from models.encoders.base_encoder import BaseEncoder


class Gamma(BaseEncoder):

    def __init__(self, input_dim=64, quantize_bits=2, noise_std=0.2, normalize=True):

        super(Gamma, self).__init__()
        self.input_dim = input_dim
        self.quantize_bits = quantize_bits
        self.noise_std = noise_std if noise_std is not None else 0
        self.normalize = normalize

        # pixelate using conv2d
        #self.proj = nn.MaxPool2d(pixelate_region, stride=pixelate_region)

        # upsize using convtranspose2d
        # self.projT = nn.ConvTranspose2d(1, 1, pixelate_region, stride=pixelate_region, bias=False)
        # self.projT.requires_grad = False
        # upsize_mat = np.ones((pixelate_region, pixelate_region))
        # self.projT.weight = nn.Parameter(torch.Tensor([[upsize_mat]]), requires_grad=False)

        # noise layer
        self.noise = nn.Conv2d(1, 1, 1, bias=False)
        self.noise.weight = nn.Parameter(torch.Tensor([[[[self.noise_std]]]]))

    def forward(self, x):
        if x.size(1) == 1:  # grayscale
            x = x.pow(10.0)
            x = x.add(self.noise(torch.randn_like(x)))
        elif x.size(1) == 3:  # RGB
            # Red
            tmpR = x[:, 0, :, :].unsqueeze(1).pow(10.0)
            # tmpR = self.proj(tmpR)
            # tmpR = tmpR.add(self.noise(torch.randn_like(tmpR)))
            # tmpR = self.projT(tmpR)
            # Green
            tmpG = x[:, 1, :, :].unsqueeze(1).pow(10.0)
            # tmpG = self.proj(tmpG)
            # tmpG = tmpG.add(self.noise(torch.randn_like(tmpG)))
            # tmpG = self.projT(tmpG)
            # Blue
            tmpB = x[:, 2, :, :].unsqueeze(1).pow(10.0)
            # tmpB = self.proj(tmpB)
            # tmpB = tmpB.add(self.noise(torch.randn_like(tmpB)))
            # tmpB = self.projT(tmpB)
            x = torch.cat((tmpR, tmpG, tmpB), dim=1)
        x = scale_max_to_1(x)
        x = quantize(x, num_bits=self.quantize_bits)
        if self.normalize:
            if x.size(1) == 1:
                x = normalize(x, mean=0.485, std=0.229)
            elif x.size(1) == 3:
                x = normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return x
