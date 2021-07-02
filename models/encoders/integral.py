"""Integral image"""

import torch.nn as nn
import torch
import skimage
import numpy as np
from models.encoders.util import quantize
from models.encoders.base_encoder import BaseEncoder


class Integral(BaseEncoder):

    def __init__(self,input_dim=64, quantize_bits=2, noise_std=0, normalize=True,gpu_idx=0):

        super(Integral, self).__init__()
        self.gpu_idx=gpu_idx
        self.quantize_bits=quantize_bits

    def forward(self, x):

        x = x.cpu().detach().numpy()
        y = np.zeros((x.shape[0],x.shape[2],x.shape[3]))
        for i in range(x.shape[0]):
            #print('Calculating for:'+str(i)+'of:'+str(x.shape[0]))
            y[i,:,:] = skimage.transform.integral.integral_image(x[i,0,:,:])
        y = y.astype('float32')
        y = torch.from_numpy(y).unsqueeze(1).cuda(self.gpu_idx)
        y = quantize(y, num_bits=self.quantize_bits)
        return y