"""Backprojection Networks

Discriminator classes that include an initial backprojection layer
The backprojection layer is meant to map a lower-dimensional vector to a larger-dimensional image
For example, if the encoder performs Ax where x is the vectorized image, a backprojection network could
first perform A^T and then perform standard classification

The discriminator class you would actually want to use should be imported
"""

import torch
import torch.nn as nn
import math
import sys
from models.discriminators.base_discriminator import BaseDiscriminator


# Appends a fully connected network ("back-projection layer") to a standard discriminator network
class BackProjectNetwork(BaseDiscriminator):

    def __init__(self, discriminator_arch=None, discriminator_params={}, proj_mat=None, num_classes=10):

        assert discriminator_arch is not None, "No discriminator arch provided in BackProjectNetowrk"

        super(BackProjectNetwork, self).__init__()
        self.discriminator = getattr(sys.modules[__name__], discriminator_arch)(num_classes=num_classes, **discriminator_params)

        proj_mat = torch.Tensor(proj_mat)
        self.imdim = proj_mat.shape[0]
        self.projdim = proj_mat.shape[1]
        self.projT = nn.Linear(self.projdim, self.imdim, bias=False)
        if proj_mat is not None:
            self.projT.weight = nn.Parameter(proj_mat)
        else:
            nn.init.normal_(self.projT.weight, 0, 0.01)  # if no proj_mat provided, initialize with random gaussian

    def forward(self, x):
        imdim_oneside = int(math.sqrt(self.imdim))
        if x.size(1) == 1:  # grayscale
            xsize = x.size()
            x = x.squeeze()
            x = self.projT(x)
            x = x.view(xsize[0], 1, imdim_oneside, imdim_oneside)
        elif x.size(1) == 3:  # RGB
            # Red
            tmpR = x[:, 0, :].view(x.size(0), -1)
            tmpR = self.projT(tmpR)
            tmpR = tmpR.view(x.size(0), 1, imdim_oneside, imdim_oneside)
            # Green
            tmpG = x[:, 1, :].view(x.size(0), -1)
            tmpG = self.projT(tmpG)
            tmpG = tmpG.view(x.size(0), 1, imdim_oneside, imdim_oneside)
            # Blue
            tmpB = x[:, 2, :].view(x.size(0), -1)
            tmpB = self.projT(tmpB)
            tmpB = tmpB.view(x.size(0), 1, imdim_oneside, imdim_oneside)
            x = torch.cat((tmpR, tmpG, tmpB), dim=1)
        x = self.discriminator(x)
        return x

    def _initialize_weights(self, proj_mat=None):
        if proj_mat is not None:
            self.projT.weight = nn.Parameter(proj_mat)
        else:
            nn.init.normal_(self.projT.weight, 0, 0.01)
        self.discriminator._initialize_weights()


# Performs a linear operation and then formats the output back into image form
# It is called a "TransposeNet" but it won't take any transposes
class TransposeNet(BaseDiscriminator):

    def __init__(self, proj_type=0, weight_init=None, indim=0, outdim=224*224, noise_var=0, bias=False):

        super(TransposeNet, self).__init__()
        self.proj_type = proj_type
        self.noise_var = noise_var
        if weight_init is not None:
            weight_init = torch.Tensor(weight_init)

        if self.proj_type == 1 or self.proj_type == 2:
            if weight_init is not None:
                indim = weight_init.shape[1]
                outdim = weight_init.shape[0]
            self.proj = nn.Linear(indim, outdim, bias=bias)
            if weight_init is not None:
                self.proj.weight = nn.Parameter(weight_init)
            else:
                nn.init.normal_(self.proj.weight, 0, 0.01)

    def forward(self, x):
        if self.proj_type != 0:
            if x.size(1) == 1:  # grayscale
                xsize = x.size()
                x = x.view(xsize[0], -1)
                x = self.proj(x)
                x = x.view(xsize[0], 1, xsize[2], xsize[3])
            elif x.size(1) == 3:  # RGB
                tmpR = x[:, 0, :, :].view(x.size(0), -1)
                tmpR = self.proj(tmpR)
                tmpR = tmpR.view(x.size(0), 1, x.size(2), x.size(3))
                tmpG = x[:, 1, :, :].view(x.size(0), -1)
                tmpG = self.proj(tmpG)
                tmpG = tmpG.view(x.size(0), 1, x.size(2), x.size(3))
                tmpB = x[:, 2, :, :].view(x.size(0), -1)
                tmpB = tmpB.view(x.size(0), 1, x.size(2), x.size(3))
                x = torch.cat((tmpR, tmpG, tmpB), dim=1)
        return x

    def _initialize_weights(self, proj_mat=None):
        if proj_mat is not None:
            self.projT.weight = nn.Parameter(proj_mat)
        else:
            nn.init.normal_(self.projT.weight, 0, 0.01)
        self.discriminator._initialize_weights()