"""Local binary patterns"""

import skimage
import torch
import torch.nn as nn
import numpy as np
from models.encoders.util import quantize
from skimage.feature import local_binary_pattern
from models.encoders.base_encoder import BaseEncoder


class LBP(BaseEncoder):

    def __init__(self,input_dim=64, quantize_bits=16, noise_std=0, normalize=True,gpu_idx=0):
        super(LBP, self).__init__()
        self.gpu_idx=gpu_idx


    def forward(self, x):
        x=x.cpu().detach().numpy()
        y=np.zeros((x.shape[0],x.shape[2],x.shape[3]))
        for i in range(x.shape[0]):
            #print('Calculating for:'+str(i)+'of:'+str(x.shape[0]))
            y[i,:,:]=local_binary_pattern(x[i,0,:,:], 8,1,'default')
        y=y.astype('float32')
        y=torch.from_numpy(y).unsqueeze(1).cuda(self.gpu_idx)
        return y