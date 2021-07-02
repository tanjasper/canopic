"""Basic Encoders

Encoders with basic operations such as adding noise
"""

import torch
import torch.nn as nn
import numpy as np
from models.encoders.util import scale_max_to_1, quantize, normalize
from scipy.ndimage import gaussian_filter
from models.encoders.base_encoder import BaseEncoder


class Normalization(BaseEncoder):
    """A class that can perform different types of normalizations

    Available modes:
        - max: divides each data piece by its max absolute value (normalizes it to be within [-1, 1])
        - imagenet: subtracts input by imagenet mean and divides by imagenet std (expects inputs to be (0, 1))
        - mean_std: subtracts input by given mean and divides by given std
        - instance_norm: performs instance norm
    """

    def __init__(self, opt, mean=None, std=None, mode=None, in_channels=None, **kwargs):
        super(Normalization, self).__init__(**kwargs)
        self.mean = mean
        self.std = std
        self.mode = mode
        if self.mode not in ['max', 'imagenet', 'mean_std', 'instance_norm']:
            raise NameError('Normalization -- unknown mode given')
        if self.mode == 'imagenet':
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        if self.mode in ['imagenet', 'mean_std']:
            if self.mean is None or self.std is None:
                raise ValueError('Normalization -- either mean or std is missing')
            self.mean = self.mean.view(1, -1, 1, 1)
            self.std = self.std.view(1, -1, 1, 1)
        if self.mode == 'instance_norm':
            self.init_instance_norm = nn.InstanceNorm2d(in_channels)

    def forward(self, x):
        if self.mode == 'max':
            return x / x.abs().view(x.size(0), -1).max(1)[0].view(-1, 1, 1, 1)
        if self.mode in ['imagenet', 'mean_std']:
            mean = self.mean
            std = self.std
            if input.is_cuda:
                mean = mean.cuda(x.get_device())
                std = std.cuda(x.get_device())
            return (x - mean) / std
        if self.mode == 'instance_norm':
            return self.init_instance_norm(x)



class PassThrough(BaseEncoder):
    """An encoder that simply lets the data pass through unaltered"""

    def __init__(self, opt=None, **kwargs):
        super(PassThrough, self).__init__(**kwargs)

    def forward(self, x):
        return x


class Noiser(BaseEncoder):

    def __init__(self, noise_std=0, learn_noise=False, quantize_bits=0, normalize=True, **kwargs):
        super(Noiser, self).__init__(**kwargs)
        self.noise_std = noise_std
        self.learn_noise = learn_noise
        self.noise = nn.Conv2d(1, 1, 1, bias=False)
        self.noise.weight = nn.Parameter(torch.Tensor([[[[self.noise_std]]]]))
        self.noise.requires_grad = True if self.learn_noise else False
        self.quantize_bits = quantize_bits
        self.normalize = normalize

    def forward(self, x):
        if x.size(1) == 1:  # grayscale
            x = x.add(self.noise(torch.randn_like(x)))
        elif x.size(1) == 3:  # RGB
            # Red
            tmpR = x[:, 0, :, :].unsqueeze(1)
            tmpR = tmpR.add(self.noise(torch.randn_like(tmpR)))
            # Green
            tmpG = x[:, 1, :, :].unsqueeze(1)
            tmpG = tmpG.add(self.noise(torch.randn_like(tmpG)))
            # Blue
            tmpB = x[:, 2, :, :].unsqueeze(1)
            tmpB = tmpB.add(self.noise(torch.randn_like(tmpB)))
            x = torch.cat((tmpR, tmpG, tmpB), dim=1)
        if self.quantize_bits:
            x = scale_max_to_1(x, clip_zero=True)
            x = quantize(x, num_bits=self.quantize_bits)
        if self.normalize:
            if x.size(1) == 1:
                x = normalize(x, mean=0.485, std=0.229)
            elif x.size(1) == 3:
                x = normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return x


class Pixelate(BaseEncoder):

    def __init__(self, input_dim=64, pixelate_region=4, quantize_bits=16, noise_std=0, normalize=True):

        super(Pixelate, self).__init__()
        self.input_dim = input_dim
        self.pixelate_region = pixelate_region
        self.quantize_bits = quantize_bits
        self.noise_std = noise_std if noise_std is not None else 0
        self.normalize = normalize

        # pixelate using conv2d
        self.proj = nn.Conv2d(1, 1, pixelate_region, stride=pixelate_region, bias=False)
        self.proj.requires_grad = False
        pixelate_mat = np.ones((pixelate_region, pixelate_region)) / (pixelate_region ** 2)
        self.proj.weight = nn.Parameter(torch.Tensor([[pixelate_mat]]), requires_grad=False)

        # upsize using convtranspose2d
        self.projT = nn.ConvTranspose2d(1, 1, pixelate_region, stride=pixelate_region, bias=False)
        self.projT.requires_grad = False
        upsize_mat = np.ones((pixelate_region, pixelate_region))
        self.projT.weight = nn.Parameter(torch.Tensor([[upsize_mat]]), requires_grad=False)

        # noise layer
        self.noise = nn.Conv2d(1, 1, 1, bias=False)
        self.noise.weight = nn.Parameter(torch.Tensor([[[[self.noise_std]]]]))

    def forward(self, x):
        if x.size(1) == 1:  # grayscale
            x = self.proj(x)
            x = x.add(self.noise(torch.randn_like(x)))
        elif x.size(1) == 3:  # RGB
            # Red
            tmpR = x[:, 0, :, :].unsqueeze(1)
            tmpR = self.proj(tmpR)
            tmpR = tmpR.add(self.noise(torch.randn_like(tmpR)))
            tmpR = self.projT(tmpR)
            # Green
            tmpG = x[:, 1, :, :].unsqueeze(1)
            tmpG = self.proj(tmpG)
            tmpG = tmpG.add(self.noise(torch.randn_like(tmpG)))
            tmpG = self.projT(tmpG)
            # Blue
            tmpB = x[:, 2, :, :].unsqueeze(1)
            tmpB = self.proj(tmpB)
            tmpB = tmpB.add(self.noise(torch.randn_like(tmpB)))
            tmpB = self.projT(tmpB)
            x = torch.cat((tmpR, tmpG, tmpB), dim=1)
        x = scale_max_to_1(x)
        x = quantize(x, num_bits=self.quantize_bits)
        if self.normalize:
            if x.size(1) == 1:
                x = normalize(x, mean=0.485, std=0.229)
            elif x.size(1) == 3:
                x = normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return x


class GaussianBlur(BaseEncoder):

    def __init__(self, opt, sigma, input_dim=64, output_dim=64, never_unfreeze=True, **kwargs):

        super(GaussianBlur, self).__init__(**kwargs)
        self.sigma = sigma
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.never_unfreeze = never_unfreeze  # keep fixed by default

        # choose padding and kernel_size to give correct dimensions in the end
        pad_dim = output_dim // 2
        kernel_size = 2*pad_dim + input_dim - output_dim + 1

        self.proj = nn.Conv2d(1, 1, kernel_size, padding=pad_dim, bias=False)
        self.proj.requires_grad = False

        # Gaussian blur kernel using numpy
        dirac = np.zeros((input_dim+1, input_dim+1))
        dirac[int(input_dim/2), int(input_dim/2)] = 1
        self.proj_mat = gaussian_filter(dirac, sigma=self.sigma)
        self.proj.weight = nn.Parameter(torch.Tensor([[self.proj_mat]]), requires_grad=False)

    def forward(self, x):
        if x.size(1) == 1:  # grayscale
            x = self.proj(x)
        elif x.size(1) == 3:  # RGB
            # Red
            tmpR = x[:, 0, :, :].unsqueeze(1)
            tmpR = self.proj(tmpR)
            # Green
            tmpG = x[:, 1, :, :].unsqueeze(1)
            tmpG = self.proj(tmpG)
            # Blue
            tmpB = x[:, 2, :, :].unsqueeze(1)
            tmpB = self.proj(tmpB)
            x = torch.cat((tmpR, tmpG, tmpB), dim=1)
        return x
