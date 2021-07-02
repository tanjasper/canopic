"""Permutation Encoders

Encoders that have to do with swapping the position of pixels
"""

import torch
import torch.nn as nn
import os
import numpy as np
from torch.nn import functional as F
from models.encoders.base_encoder import BaseEncoder


class Perm(BaseEncoder):
    """

    TODO:
        Set kernel_size to be a required parameter
        Set imsize to be according to opt?
    """
    def __init__(self,  opt, kernel_size=11, num_filters=1, perm_mat_path=None, imsize=64, red_dim=16):
        super(Perm, self).__init__()
        self.red_dim = red_dim
        self.imsize = imsize
        if perm_mat_path:
            print('Perm -- loading permutation matrix from ' + os.path.join(opt.root_data_dir, perm_mat_path))
            kernel = np.load(os.path.join(opt.root_data_dir, perm_mat_path))
        else:
            kernel = None
        self.mat = nn.Parameter(torch.randn((int((self.imsize/self.red_dim)**2),int(self.imsize**2))))
        self.proj = nn.Conv2d(1, num_filters, kernel_size, bias=False)
        if kernel is not None:
            if num_filters == 1:
                if kernel.ndim == 2 and kernel.shape == (kernel_size, kernel_size):
                    self.proj.weight = nn.Parameter(torch.Tensor(kernel[np.newaxis, np.newaxis, :]))
                elif kernel.ndim == 3 and kernel.shape == (1, kernel_size, kernel_size):
                    self.proj.weight = nn.Parameter(torch.Tensor(kernel[:, np.newaxis, :, :]))
                elif kernel.ndim == 4 and kernel.shape == (1, 1, kernel_size, kernel_size):
                    self.proj.weight = nn.Parameter(torch.Tensor(kernel))
                else:
                    raise Exception('provided kernel dimensions do not match NN dimensions')
            elif num_filters > 1:
                if kernel.ndim == 3 and kernel.shape == (num_filters, kernel_size, kernel_size):
                    self.proj.weight = nn.Parameter(torch.Tensor(kernel[:, np.newaxis, :, :]))
                elif kernel.ndim == 4 and kernel.shape == (num_filters, 1, kernel_size, kernel_size):
                    self.proj.weight = nn.Parameter(torch.Tensor(kernel))
                else:
                    raise Exception('provided kernel dimensions do not match NN dimensions')
        self.proj.weight.requires_grad = False

    def forward(self, x):
        if x.size(1) == 1:  # grayscale
            x = self.proj(x)
        elif x.size(1) == 3:      # RGB
            x = 0.5*(x[:, 0, :, :]+x[:, 1, :, :]+x[:, 2, :, :]).unsqueeze(1)
            x = self.proj(x)
        self.mat.data = torch.clamp(self.mat.data,0)
        self.mat.data = self.mat.data/(self.mat.data.sum(dim=1, keepdim=True)+1e-3)
        y = F.linear(x.view(-1,int(self.imsize**2)),self.mat,bias=None).view(-1,1,int(self.imsize/self.red_dim),int(self.imsize/self.red_dim))
        return y

    def save_debug_data(self, save_dir, savename_suffix=''):
        savename = 'perm_mat_' + savename_suffix
        np.save(os.path.join(save_dir, savename), self.mat.detach().cpu().numpy())
