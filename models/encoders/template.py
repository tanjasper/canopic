"""Template-Based Encoders"""

import scipy.io as sio
import torch
import torch.nn as nn
import numpy as np
from models.encoders.util import normalize, scale_max_to_1, quantize
from models.encoders.base_encoder import BaseEncoder


class FaceTemp(BaseEncoder):

    def __init__(self, frequency=1, theta=0, n_stds=5,input_dim=64, quantize_bits=16, noise_std=0, normalize=True):

        super(FaceTemp, self).__init__()
        # self.frequency = frequency
        # self.theta = theta
        # self.n_stds = n_stds
        self.quantize_bits = quantize_bits
        self.noise_std = noise_std if noise_std is not None else 0
        self.normalize = normalize

        self.proj = nn.Conv2d(1, 1, input_dim + 1, padding=int(input_dim / 2), bias=False)
        self.proj.requires_grad = False

        # Gaussian blur kernel using numpy
        # dirac = np.zeros((input_dim + 1, input_dim + 1))
        # dirac[int(input_dim / 2), int(input_dim / 2)] = 1
        #gab = sf.gabor_kernel(frequency=self.frequency, theta=self.theta, n_stds=self.n_stds).real
        temp = sio.loadmat('average_1000_vggface2.mat')['sm']
        template = 0.67*(temp[:,:,0]+temp[:,:,1]+temp[:,:,2])
        self.proj_mat = np.pad(template,[1,0],'constant')

        self.proj.weight = nn.Parameter(torch.Tensor([[self.proj_mat]]), requires_grad=False)

        # noise layer
        self.noise = nn.Conv2d(1, 1, 1, bias=False)
        self.noise.weight = nn.Parameter(torch.Tensor([[[[self.noise_std]]]]))

    def forward(self, x):
        if x.size(1) == 1:  # grayscale
            x = self.proj(x)
            x = x.add(self.noise(torch.rand_like(x)))
        elif x.size(1) == 3:  # RGB
            # Red
            tmpR = x[:, 0, :, :].unsqueeze(1)
            tmpR = self.proj(tmpR)
            tmpR = tmpR.add(self.noise(torch.randn_like(tmpR)))
            # Green
            tmpG = x[:, 1, :, :].unsqueeze(1)
            tmpG = self.proj(tmpG)
            tmpG = tmpG.add(self.noise(torch.randn_like(tmpG)))
            # Blue
            tmpB = x[:, 2, :, :].unsqueeze(1)
            tmpB = self.proj(tmpB)
            tmpB = tmpB.add(self.noise(torch.randn_like(tmpB)))
            x = torch.cat((tmpR, tmpG, tmpB), dim=1)
        x = scale_max_to_1(x)
        x = quantize(x, num_bits=self.quantize_bits)
        if self.normalize:
            if x.size(1) == 1:
                x = normalize(x, mean=0.485, std=0.229)
            elif x.size(1) == 3:
                x = normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return x