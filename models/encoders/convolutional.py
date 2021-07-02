"""Convolutional encoders

This module contains classes that perform 2D convolutions on the RGB image.
The convolution is performed individually for each color channel to return an RGB image.
"""

import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from models.encoders.util import normalize, scale_max_to_1, quantize
from models.encoders.base_encoder import BaseEncoder


class Convolution(BaseEncoder):
    """2D channelwise convolutional encoder

    This is the convolutional encoder used for WACV experiments.
    Note that the same kernel is applied to each input channel. This mimics a convolutional optical mask,
        wherein (approximately) the same mask convolution is performed on each color individually.

    Settings:
        kernel_size (int): dimension of the convolutional kernels
        num_filters (int): number of convolutional filters to apply
        kernel_path (str): path of numpy arrays for the initial weights of the kernels. Leave unspecified
            to initialize kernels with Pytorch's default initialization method.
        """

    def __init__(self, opt, kernel_size, num_filters, padding=0, padmode='constant', kernel_path=None, **kwargs):

        super(Convolution, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.padding = padding
        if type(self.padding) is int:
            self.padding = (self.padding, self.padding, self.padding, self.padding)
        self.padmode = padmode
        if kernel_path and not self.skip_init_loading:
            print('Convolution -- loading kernel from ' + os.path.join(opt.root_data_dir, kernel_path))
            kernel = np.load(os.path.join(opt.root_data_dir, kernel_path))
        else:
            kernel = None
        # doing Pytorch's default random initialization: kaiming_uniform_(a=sqrt(5))
        # See _ConvNd class: https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
        self.proj = nn.Conv2d(1, num_filters, kernel_size, padding=0, bias=False)

        # set convolution kernel
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

    def forward(self, x):
        # self.proj.weight.data = torch.clamp(self.proj.weight.data, 0)
        # self.proj.weight.data = F.normalize(self.proj.weight.data)

        if self.padding:
            x = F.pad(x, self.padding, mode=self.padmode)
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

    def save_debug_data(self, save_dir, savename_suffix=''):
        savename = 'convolution_kernel_' + savename_suffix
        np.save(os.path.join(save_dir, savename), self.proj.weight.detach().cpu().numpy())


class ConvGenerator(BaseEncoder):
    """Performs a convolution and potentially max-pooling, average pooling, quantization

    User parameters:
        learn_noise (bool)
        kernel_size (int)
        num_filters (int)
        learn_noise (bool)
        mxp (bool): Whether to perform maxpool or not
        avg (bool): Whether to perform average pooling or not
        quantize (bool): Whether to perform quantization or not
        quantize_bits (int): How many bits to use for quantization
        mode (string?)
        red_dim (?)

    TODO:
        set kernel_size to be a required parameter
    """

    def __init__(self, opt, kernel_size=11, num_filters=1, kernel_path=None, learn_noise=True, mxp=False, quantize_bits=4,
                 quantize=True, avg=False, mode='bilinear', red_dim=16):

        super(ConvGenerator, self).__init__()
        # Set parameters
        self.learn_noise = learn_noise
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.mxp = mxp
        self.avg = avg
        self.avgpool = nn.AvgPool2d(16, 16)
        self.quantize = quantize
        self.bits = quantize_bits
        self.mode = mode
        self.red_dim = red_dim
        if kernel_path:
            print('ConvGenerator -- loading kernel from ' + os.path.join(opt.root_data_dir, opt.proj_mat_init_path))
            kernel = np.load(os.path.join(opt.root_data_dir, opt.proj_mat_init_path))
        else:
            kernel = None

        # Pytorch objects
        self.noise = nn.Parameter(torch.Tensor([0])).type(torch.Tensor)
        self.noise.requires_grad = True if self.learn_noise else False
        self.mxpool = nn.MaxPool2d(int(red_dim), int(red_dim))

        # doing Pytorch's default random initialization: kaiming_uniform_(a=sqrt(5))
        # See _ConvNd class: https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
        self.proj = nn.Conv2d(1, num_filters, kernel_size, bias=False)

        # set convolution kernel
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

    def forward(self, x):

        if x.size(1) == 1:  # grayscale
            x = self.proj(x)
        elif x.size(1) == 3:  # RGB
            tmpR = x[:, 0, :, :].unsqueeze(1)
            tmpR = self.proj(tmpR)
            tmpG = x[:, 1, :, :].unsqueeze(1)
            tmpG = self.proj(tmpG)
            tmpB = x[:, 2, :, :].unsqueeze(1)
            tmpB = self.proj(tmpB)
            x = torch.cat((tmpR, tmpG, tmpB), dim=1)
            imsize = x.shape[2]
            factor = (imsize / float(self.red_dim))
        if self.mxp:
            x = self.mxpool(x)
        elif self.avg:
            x = F.interpolate(x, scale_factor=1 / (factor), mode='bilinear')
        x = x.add(torch.mul(torch.randn_like(x), self.noise))  # Noise
        if self.quantize:
            qmin = 0.
            qmax = 2. ** self.bits - 1.
            min_value = x.min()
            max_value = x.max()
            scale = (max_value - min_value) / (qmax - qmin)
            scale = max(scale, 1e-8)
            x = x.add(-min_value).div(scale).add(qmin)

            ### quantize
            # x.clamp_(qmin, qmax).round_()  # quantize
            x = x.clamp(qmin, qmax)
            y = x.detach()  # detach to round and prop gradient through round
            y = y - y.round()
            x = x - y
            x = x.add(-qmin).mul(scale).add(min_value)  # dequantize
        if self.avg or self.mxp:
            x = F.interpolate(x, scale_factor=int(self.red_dim), mode=self.mode)
        return x


class ConvWithFlexibleTranspose(BaseEncoder):

    def __init__(self, opt, proj_type=0, kernel_path=None, noise_std=0, learn_noise=False):

        super(ConvWithFlexibleTranspose, self).__init__()
        self.proj_type = proj_type
        self.noise_std = noise_std
        self.learn_noise = learn_noise
        if kernel_path:
            print('ConvWithForcedTranspose -- loading kernel from ' + os.path.join(opt.root_data_dir, kernel_path))
            kernel = np.load(os.path.join(opt.root_data_dir, kernel_path))
        else:
            kernel_size = int(opt.input_dim / 2)
            norm_mean = 0
            norm_std = 1 / kernel_size
            kernel = np.random.normal(norm_mean, norm_std, (kernel_size, kernel_size))
        kernel_size = kernel.shape[0]
        kernel = torch.Tensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_flip = torch.Tensor(torch.flip(kernel, dims=[2, 3]))

        if self.proj_type == 1 or self.proj_type == 2:
            self.proj = nn.Conv2d(1, 1, kernel_size, padding=kernel_size-1)
            self.projT = nn.Conv2d(1, 1, kernel_size)
            self.proj.weight = nn.Parameter(kernel)
            self.projT.weight = nn.Parameter(kernel_flip)
            self.proj.weight.requires_grad = False  # fix these matrices
            if self.proj_type == 1:
                self.projT.weight.requires_grad = False
            else:
                self.projT.weight.requires_grad = True  # allow projT to be learned
        self.noise = nn.Conv2d(1, 1, 1, bias=False)
        self.noise.weight = nn.Parameter(torch.Tensor([[[[self.noise_std]]]]))
        self.noise.requires_grad = True if self.learn_noise else False

    def forward(self, x):
        if self.proj_type != 0:
            if x.size(1) == 1:  # grayscale
                x = self.proj(x)
                x = x.add(self.noise(torch.randn_like(x)))
                x = self.projT(x)
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
        return x


def ConvWithFixedFlexibleTranspose(opt, kernel=None, noise_std=0, learn_noise=False):

    return ConvWithFlexibleTranspose(opt, proj_type=1, kernel=kernel, noise_std=noise_std, learn_noise=learn_noise)


def ConvWithLearnedTranspose(opt, kernel=None, noise_std=0, learn_noise=False):

    return ConvWithFlexibleTranspose(opt, proj_type=2, kernel=kernel, noise_std=noise_std, learn_noise=learn_noise)


# A network that performs a convolution and then its transpose.
# The transpose is forced to be transpose of convolution -- they are trained together
class ConvWithForcedTranspose(BaseEncoder):

    def __init__(self, opt, kernel_size=32, kernel_path=None, noise_std=0, learn_noise=False):

        super(ConvWithForcedTranspose, self).__init__()
        self.kernel_size = kernel_size
        self.learn_noise = learn_noise
        self.noise_std = noise_std
        if kernel_path:
            print('ConvWithForcedTranspose -- loading kernel from ' + os.path.join(opt.root_data_dir, kernel_path))
            kernel = np.load(os.path.join(opt.root_data_dir, kernel_path))
            self.kernel_size = kernel.shape[0]
        else:
            norm_mean = 0
            norm_std = 1 / self.kernel_size
            kernel = np.random.normal(norm_mean, norm_std, (kernel_size, kernel_size))

        # one layer that performs linear projection then its transpose
        self.proj = ConvWithTransposeLayer(self.kernel_size, init_weight=kernel,
                                             noise_std=self.noise_std, learn_noise=self.learn_noise)

    # pass x through the one layer with proper resizing
    def forward(self, x):
        if x.size(1) == 1:  # grayscale
            x = self.proj(x)
        elif x.size(1) == 3:  # RGB
            tmpR = x[:, 0, :, :].unsqueeze(1)
            tmpR = self.proj(tmpR)
            tmpG = x[:, 1, :, :].unsqueeze(1)
            tmpG = self.proj(tmpG)
            tmpB = x[:, 2, :, :].unsqueeze(1)
            tmpB = self.proj(tmpB)
            x = torch.cat((tmpR, tmpG, tmpB), dim=1)
        return x


# a fully connected layer that would also perform its transpose
class ConvWithTransposeLayer(BaseEncoder):

    def __init__(self, opt, kernel_size, bias=False, init_weight=None, noise_std=0, learn_noise=False):
        super(ConvWithTransposeLayer, self).__init__()
        if init_weight is not None:
            assert init_weight.shape[0] == kernel_size and init_weight.shape[1] == kernel_size, \
                "Provided initial weights do not match provided kernel_size"
        self.kernel_size = kernel_size
        self.init_weight = init_weight
        self.noise_std = noise_std
        self.has_bias = bias
        self.learn_noise = learn_noise
        self.weight = nn.Parameter(torch.Tensor(1, 1, kernel_size, kernel_size))
        if bias:
            self.bias_proj = nn.Parameter(torch.Tensor(1))
            self.bias_t = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('bias_proj', None)
            self.register_parameter('bias_t', None)
        # Noise
        self.noise = nn.Parameter(torch.Tensor([[[[self.noise_std]]]]))
        self.noise.requires_grad = True if self.learn_noise else False
        # Initialize weight
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weight is not None:
            self.weight = nn.Parameter(torch.Tensor(self.init_weight).unsqueeze(0).unsqueeze(0))  # conv2d weight needs to have 4 dimensions... check documentation https://pytorch.org/docs/stable/_modules/torch/nn/functional.html
        else:
            nn.init.normal_(self.weight, 0, 1 / self.kernel_size)
        if self.has_bias:
            fan_in_proj, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound_proj = 1 / math.sqrt(fan_in_proj)
            fan_in_t, _ = nn.init._calculate_fan_in_and_fan_out(self.weight.t())
            bound_t = 1 / math.sqrt(fan_in_proj)
            nn.init.uniform_(self.bias_proj, -bound_proj, bound_proj)
            nn.init.uniform_(self.bias_t, -bound_t, bound_t)

    def forward(self, input):
        # project
        x = F.conv2d(input, self.weight, self.bias_proj, padding=self.kernel_size-1)
        # add noise
        x = x.add(F.conv2d(torch.randn_like(x), self.noise))
        x = F.conv2d(x, torch.flip(self.weight, dims=[2, 3]), self.bias_t, padding=0)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias_proj={}, bias_t={}'.format(
            self.in_features, self.out_features, self.bias_proj, self.bias_t is not None
        )


class Pretrained_Filters(BaseEncoder):

    def __init__(self,input_dim=64, quantize_bits=2, noise_std=0, normalize=False):

        super(Pretrained_Filters, self).__init__()

        self.input_dim = input_dim
        self.quantize_bits = quantize_bits
        self.noise_std = noise_std if noise_std is not None else 0
        self.normalize = normalize

        self.proj1 = nn.Conv2d(64, 1, 7, padding=3, bias=False)
        self.proj1.requires_grad = False

        self.proj2 = nn.Conv2d(64, 1, 7, padding=3, bias=False)
        self.proj2.requires_grad = False

        self.proj3 = nn.Conv2d(64, 1, 7, padding=3, bias=False)
        self.proj3.requires_grad = False

        filt_bank = (np.load('filter_resnet18_ILSVRC.npy'))
        self.proj1.weight = nn.Parameter(torch.Tensor(filt_bank[:, 0, :, :]).unsqueeze(1), requires_grad=False)
        self.proj2.weight = nn.Parameter(torch.Tensor(filt_bank[:, 1, :, :]).unsqueeze(1), requires_grad=False)
        self.proj3.weight = nn.Parameter(torch.Tensor(filt_bank[:, 2, :, :]).unsqueeze(1), requires_grad=False)

        # noise layer
        self.noise = nn.Conv2d(64, 64, 1, bias=False)
        self.noise.weight = nn.Parameter(torch.Tensor([[[[self.noise_std]]]]))

    def forward(self, x):
        if x.size(1) == 1:  # grayscale
            x = self.proj1(x)
            x = x.add(self.noise(torch.rand_like(x)))
        elif x.size(1) == 3:  # RGB
            # Red
            tmpR = x[:, 0, :, :].unsqueeze(1)
            tmpR = self.proj1(tmpR)
            #tmpR = tmpR.add(self.noise(torch.randn_like(tmpR)))
            # Green
            tmpG = x[:, 1, :, :].unsqueeze(1)
            tmpG = self.proj2(tmpG)
            #tmpG = tmpG.add(self.noise(torch.randn_like(tmpG)))
            # Blue
            tmpB = x[:, 2, :, :].unsqueeze(1)
            tmpB = self.proj3(tmpB)
            #tmpB = tmpB.add(self.noise(torch.randn_like(tmpB)))
            x = torch.cat((tmpR, tmpG, tmpB), dim=1)
        x = scale_max_to_1(x,with_negative=True)
        x = quantize(x, num_bits=self.quantize_bits)
        if self.normalize:
            if x.size(1) == 1:
                x = normalize(x, mean=0.485, std=0.229)
            elif x.size(1) == 3:
                x = normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return x


class Pretrained_Filters2(BaseEncoder):

    def __init__(self,input_dim=64, quantize_bits=2, noise_std=0, normalize=False):

        super(Pretrained_Filters2, self).__init__()

        self.input_dim = input_dim
        self.quantize_bits = quantize_bits
        self.noise_std = noise_std if noise_std is not None else 0
        self.normalize = normalize

        self.proj1 = nn.Conv2d(64, 1, 3, padding=1, bias=False)
        self.proj1.requires_grad = False

        self.proj2 = nn.Conv2d(64, 1, 3, padding=1, bias=False)
        self.proj2.requires_grad = False

        self.proj3 = nn.Conv2d(64, 1, 3, padding=1, bias=False)
        self.proj3.requires_grad = False

        filt_bank = np.load('ssd_filt_bank.npy')
        self.proj1.weight = nn.Parameter(torch.Tensor(filt_bank[:, 0, :, :]).unsqueeze(1), requires_grad=False)
        self.proj2.weight = nn.Parameter(torch.Tensor(filt_bank[:, 1, :, :]).unsqueeze(1), requires_grad=False)
        self.proj3.weight = nn.Parameter(torch.Tensor(filt_bank[:, 2, :, :]).unsqueeze(1), requires_grad=False)

        # noise layer
        self.noise = nn.Conv2d(64, 64, 1, bias=False)
        self.noise.weight = nn.Parameter(torch.Tensor([[[[self.noise_std]]]]))

    def forward(self, x):
        if x.size(1) == 1:  # grayscale
            x = self.proj1(x)
            x = x.add(self.noise(torch.rand_like(x)))
        elif x.size(1) == 3:  # RGB
            # Red
            tmpR = x[:, 0, :, :].unsqueeze(1)
            tmpR = self.proj1(tmpR)
            #tmpR = tmpR.add(self.noise(torch.randn_like(tmpR)))
            # Green
            tmpG = x[:, 1, :, :].unsqueeze(1)
            tmpG = self.proj2(tmpG)
            #tmpG = tmpG.add(self.noise(torch.randn_like(tmpG)))
            # Blue
            tmpB = x[:, 2, :, :].unsqueeze(1)
            tmpB = self.proj3(tmpB)
            #tmpB = tmpB.add(self.noise(torch.randn_like(tmpB)))
            x = torch.cat((tmpR, tmpG, tmpB), dim=1)
        x = scale_max_to_1(x,with_negative=True)
        x = quantize(x, num_bits=self.quantize_bits)
        if self.normalize:
            if x.size(1) == 1:
                x = normalize(x, mean=0.485, std=0.229)
            elif x.size(1) == 3:
                x = normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return x