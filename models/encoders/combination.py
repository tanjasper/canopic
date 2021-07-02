"""Combination Encoders

This script contains encoders that are combinations of multiple encoders in series.
Ideally, a class would be written that can take in multiple individual encoders, so that one
would not need to write a new combination encoder for every combination. This is in works.
In the meantime, this module will hold all combination encoders.
"""

import torch
import torch.nn as nn
import os
from torch.nn import functional as F
import numpy as np
from models.encoders.base_encoder import BaseEncoder

def heaviside(x):
    a = x>=0
    a = a.float()
    return a

class ConvPermQuantize(BaseEncoder):
    """
    TODO:
        set kernel_size to be required
        set imsize according to opt
    """

    def __init__(self, opt, kernel_size=11, num_filters=4, imsize=64, kernel_path=None, mxp=False, quantize_bits=4,
                 quantize=True, avg=False, mode='bilinear', red_dim=16):

        super(ConvPermQuantize, self).__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.bits = quantize_bits
        self.mode = mode
        self.red_dim = red_dim
        self.imsize = imsize
        if kernel_path:
            print('ConvPermQuantize -- loading kernel from ' + os.path.join(opt.root_data_dir, kernel_path))
            kernel = np.load(os.path.join(opt.root_data_dir, kernel_path))
        else:
            kernel = None
        self.mat = nn.Parameter(torch.randn((int((self.imsize / self.red_dim) ** 2), int(self.imsize ** 2))))
        self.bias = nn.Parameter(
            torch.from_numpy(np.linspace(0, 2 ** self.bits - 1, 2 ** self.bits, dtype='float32')[:-1] + 0.5))
        self.levels = np.linspace(0, 2 ** self.bits - 1, 2 ** self.bits, dtype='float32')[:-1]
        # self.scale = nn.Parameter(torch.tensor([1],dtype=torch.float))

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
        # print(self.red_dim)
        self.mat.data = torch.clamp(self.mat.data, 0)
        self.mat.data = self.mat.data / (self.mat.data.sum(dim=1, keepdim=True) + 1e-3)
        # print(x.shape)
        # print(self.noise)
        z = torch.zeros(x.shape[0], int(x.shape[1] * self.num_filters), int(self.imsize / self.red_dim),
                        int(self.imsize / self.red_dim), device=x.device)
        if x.size(1) == 1:  # grayscale
            x = self.proj(x)
        # x = F.linear(x.view(-1,int(self.imsize**2)),self.mat,bias=None).view(-1,1,int(self.imsize/self.red_dim),int(self.imsize/self.red_dim))
        elif x.size(1) == 3:  # RGB
            # Red
            tmpR = x[:, 0, :, :].unsqueeze(1)
            tmpR = self.proj(tmpR)
            # tmpR = F.linear(tmpR.view(-1,int(self.imsize**2)),self.mat,bias=None).view(-1,1,int(self.imsize/self.red_dim),int(self.imsize/self.red_dim))
            # Green
            tmpG = x[:, 1, :, :].unsqueeze(1)
            tmpG = self.proj(tmpG)
            # tmpG = F.linear(tmpG.view(-1,int(self.imsize**2)),self.mat,bias=None).view(-1,1,int(self.imsize/self.red_dim),int(self.imsize/self.red_dim))
            # Blue
            tmpB = x[:, 2, :, :].unsqueeze(1)
            tmpB = self.proj(tmpB)
            # tmpB = F.linear(tmpB.view(-1,int(self.imsize**2)),self.mat,bias=None).view(-1,1,int(self.imsize/self.red_dim),int(self.imsize/self.red_dim))
            x = torch.cat((tmpR, tmpG, tmpB), dim=1)
        # print(x.shape)
        for i in range(int(x.shape[1])):
            tmp = F.linear(x[:, i, :, :].unsqueeze(1).view(-1, int(self.imsize ** 2)), self.mat, bias=None)
            # print(tmp.shape)
            z[:, i, :, :] = tmp.view(-1, int(self.imsize / self.red_dim), int(self.imsize / self.red_dim))
        # print(z.shape)

        qmin = 0.
        qmax = 2. ** self.bits - 1.
        min_value = z.min()
        max_value = z.max()
        scale_value = (max_value - min_value) / (qmax - qmin)
        scale_value = max(scale_value, 1e-4)
        z = ((z - min_value) / ((max_value - min_value) + 1e-4)) * (qmax - qmin)
        # print(x.max())
        y = torch.zeros(z.shape, device=x.device)
        # dummyWeight = self.bias.data.clamp(0,(2**self.bits-1))
        # self.bias.data = dummyWeight.sort(0).values
        self.bias.data = self.bias.data.clamp(0, (2 ** self.bits - 1))
        self.bias.data = self.bias.data.sort(0).values
        for i in range(self.levels.shape[0]):
            y = y + torch.sigmoid(5 * ((z) - self.bias[i]))
        # print(y.max())
        # print(self.bias.data.max())
        # print(y.requires_grad)
        y = y.mul(scale_value).add(min_value)
        # print(y.shape)
        # print(y.max())
        y = F.interpolate(y, scale_factor=int(self.red_dim), mode=self.mode)
        # print(self.bias.grad)
        return y

    def save_debug_data(self, save_dir, savename_suffix=''):
        savename = 'perm_mat_' + savename_suffix
        np.save(os.path.join(save_dir, savename), self.mat.detach().cpu().numpy())
        savename = 'kernel_' + savename_suffix
        np.save(os.path.join(save_dir, savename), self.proj.weight.detach().cpu().numpy())
        savename = 'bias_' + savename_suffix
        np.save(os.path.join(save_dir, savename), self.bias.detach().cpu().numpy())


class ConvMxPQuantize(BaseEncoder):

    def __init__(self, opt, kernel_size=11, sig_scale=5, num_filters=4, kernel_path=None, learn_noise=True, mxp=False,
                 quantize_bits=4, exact_quantize=False, mode='bilinear', red_dim=16, padding=0, pool_padding=0, **kwargs):

        super(ConvMxPQuantize, self).__init__(**kwargs)
        self.exact_quantize = exact_quantize
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.bits = quantize_bits
        self.mode = mode
        self.red_dim = red_dim
        self.mxpool = nn.MaxPool2d(int(red_dim), int(red_dim), padding=pool_padding)
        self.sig_scale = sig_scale
        if kernel_path and not self.skip_init_loading:
            print('ConvMxPQuantize -- loading kernel from ' + os.path.join(opt.root_data_dir, kernel_path))
            kernel = np.load(os.path.join(opt.root_data_dir, kernel_path))
        else:
            kernel = None
        self.bias = nn.Parameter(
            torch.from_numpy(np.linspace(0, 2 ** self.bits - 1, 2 ** self.bits, dtype='float32')[:-1] + 0.5))
        self.levels = np.linspace(0, 2 ** self.bits - 1, 2 ** self.bits, dtype='float32')[:-1]
        # self.scale = nn.Parameter(torch.tensor([1],dtype=torch.float))

        # doing Pytorch's default random initialization: kaiming_uniform_(a=sqrt(5))
        # See _ConvNd class: https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
        self.proj = nn.Conv2d(1, num_filters, kernel_size, padding=padding, bias=False)

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

        # print(x.shape)
        # print(self.noise)
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
        x = self.mxpool(x)
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
        else:
            qmin = 0.
            qmax = 2. ** self.bits - 1.
            min_value = x.view(x.size(0), -1).min(1)[0].view(-1, 1, 1, 1)  # min values for each input
            max_value = x.view(x.size(0), -1).max(1)[0].view(-1, 1, 1, 1)  # max values for each input
            scale_value = (max_value - min_value) / (qmax - qmin)
            scale_value = torch.clamp(scale_value, 1e-4)
            x = ((x - min_value) / ((max_value - min_value) + 1e-4)) * (qmax - qmin)
            # print(x.max())
            y = torch.zeros(x.shape, device=x.device)
            # dummyWeight = self.bias.data.clamp(0,(2**self.bits-1))
            # self.bias.data = dummyWeight.sort(0).values
            self.bias.data = self.bias.data.clamp(0, (2 ** self.bits - 1))
            self.bias.data = self.bias.data.sort(0)[0]  # for more recent versions of pytorch: self.bias.data = self.bias.data.sort(0).values
            for i in range(self.levels.shape[0]):
                y = y + torch.sigmoid(self.sig_scale * ((x) - self.bias[i]))
            # print(y.max())
            # print(self.bias.data.max())
            # print(y.requires_grad)
            y = y.mul(scale_value).add(min_value)
        # print(y.shape)
        # print(y.max())
        y = F.interpolate(y, scale_factor=int(self.red_dim), mode=self.mode)
        # print(y.shape)
        return y

    def save_debug_data(self, save_dir, savename_suffix=''):
        savename = 'bias_' + savename_suffix
        np.save(os.path.join(save_dir, savename), self.bias.detach().cpu().numpy())
        savename = 'kernel_' + savename_suffix
        np.save(os.path.join(save_dir, savename), self.proj.weight.detach().cpu().numpy())


class PosConvMxPQuantize(BaseEncoder):

    def __init__(self, opt, kernel_size=11, sig_scale=5, num_filters=4, kernel_path=None, learn_noise=True, mxp=False,
                 quantize_bits=4, quantize=True, avg=False, mode='bilinear', red_dim=16):

        super(PosConvMxPQuantize, self).__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.bits = quantize_bits
        self.mode = mode
        self.red_dim = red_dim
        self.mxpool = nn.MaxPool2d(int(red_dim), int(red_dim))
        self.sig_scale = sig_scale
        self.bias = nn.Parameter(
            torch.from_numpy(np.linspace(0, 2 ** self.bits - 1, 2 ** self.bits, dtype='float32')[:-1] + 0.5))
        self.levels = np.linspace(0, 2 ** self.bits - 1, 2 ** self.bits, dtype='float32')[:-1]
        if kernel_path:
            print('PosconvMxPQuantize -- loading kernel from ' + os.path.join(opt.root_data_dir, kernel_path))
            kernel = np.load(os.path.join(opt.root_data_dir, kernel_path))
        else:
            kernel = None
        # self.scale = nn.Parameter(torch.tensor([1],dtype=torch.float))

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

        # print(x.shape)
        # print(self.noise)
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
        x = self.mxpool(x)
        qmin = 0.
        qmax = 2. ** self.bits - 1.
        min_value = x.min()
        max_value = x.max()
        scale_value = (max_value - min_value) / (qmax - qmin)
        scale_value = max(scale_value, 1e-4)
        x = ((x - min_value) / ((max_value - min_value) + 1e-4)) * (qmax - qmin)
        # print(x.max())
        y = torch.zeros(x.shape, device=x.device)
        # dummyWeight = self.bias.data.clamp(0,(2**self.bits-1))
        # self.bias.data = dummyWeight.sort(0).values
        self.proj.weight.data = torch.clamp(self.proj.weight.data, 0)
        self.bias.data = self.bias.data.clamp(0, (2 ** self.bits - 1))
        self.bias.data = self.bias.data.sort(0).values
        for i in range(self.levels.shape[0]):
            y = y + torch.sigmoid(self.sig_scale * ((x) - self.bias[i]))
        # print(y.max())
        # print(self.bias.data.max())
        # print(y.requires_grad)
        y = y.mul(scale_value).add(min_value)
        # print(y.shape)
        # print(y.max())
        y = F.interpolate(y, scale_factor=int(self.red_dim), mode=self.mode)
        # print(y.shape)
        return y

    def save_debug_data(self, save_dir, savename_suffix=''):
        savename = 'bias_' + savename_suffix
        np.save(os.path.join(save_dir, savename), self.bias.detach().cpu().numpy())
        savename = 'kernel_' + savename_suffix
        np.save(os.path.join(save_dir, savename), self.proj.weight.detach().cpu().numpy())


class ConvPerm(BaseEncoder):

    def __init__(self, kernel_size=11, num_filters=4, imsize=64, kernel=None, mxp=False, quantize=True, avg=False,
                 mode='bilinear', red_dim=16):

        super(ConvPerm, self).__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        # self.bits = quantize_bits
        self.mode = mode
        self.red_dim = red_dim
        self.imsize = imsize
        self.mat = nn.Parameter(torch.randn((int((self.imsize / self.red_dim) ** 2), int(self.imsize ** 2))))
        # self.bias = nn.Parameter(torch.from_numpy(np.linspace(0,2**self.bits-1,2**self.bits,dtype='float32')[:-1]+0.5))
        # self.levels = np.linspace(0,2**self.bits-1,2**self.bits,dtype='float32')[:-1]
        # self.scale = nn.Parameter(torch.tensor([1],dtype=torch.float))

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
        # print(self.red_dim)
        self.mat.data = torch.clamp(self.mat.data, 0)
        self.mat.data = self.mat.data / (self.mat.data.sum(dim=1, keepdim=True) + 1e-3)
        # print(x.shape)
        # print(self.noise)
        z = torch.zeros(x.shape[0], int(x.shape[1] * self.num_filters), int(self.imsize / self.red_dim),
                        int(self.imsize / self.red_dim), device=x.device)
        if x.size(1) == 1:  # grayscale
            x = self.proj(x)
        # x = F.linear(x.view(-1,int(self.imsize**2)),self.mat,bias=None).view(-1,1,int(self.imsize/self.red_dim),int(self.imsize/self.red_dim))
        elif x.size(1) == 3:  # RGB
            # Red
            tmpR = x[:, 0, :, :].unsqueeze(1)
            tmpR = self.proj(tmpR)
            # tmpR = F.linear(tmpR.view(-1,int(self.imsize**2)),self.mat,bias=None).view(-1,1,int(self.imsize/self.red_dim),int(self.imsize/self.red_dim))
            # Green
            tmpG = x[:, 1, :, :].unsqueeze(1)
            tmpG = self.proj(tmpG)
            # tmpG = F.linear(tmpG.view(-1,int(self.imsize**2)),self.mat,bias=None).view(-1,1,int(self.imsize/self.red_dim),int(self.imsize/self.red_dim))
            # Blue
            tmpB = x[:, 2, :, :].unsqueeze(1)
            tmpB = self.proj(tmpB)
            # tmpB = F.linear(tmpB.view(-1,int(self.imsize**2)),self.mat,bias=None).view(-1,1,int(self.imsize/self.red_dim),int(self.imsize/self.red_dim))
            x = torch.cat((tmpR, tmpG, tmpB), dim=1)
        # print(x.shape)
        for i in range(int(x.shape[1])):
            tmp = F.linear(x[:, i, :, :].unsqueeze(1).view(-1, int(self.imsize ** 2)), self.mat, bias=None)
            # print(tmp.shape)
            z[:, i, :, :] = tmp.view(-1, int(self.imsize / self.red_dim), int(self.imsize / self.red_dim))
        # print(z.max())
        z = F.interpolate(z, scale_factor=int(self.red_dim), mode=self.mode)
        # print(z.shape)
        return z

    def save_debug_data(self, save_dir, savename_suffix=''):
        savename = 'perm_mat_' + savename_suffix
        np.save(os.path.join(save_dir, savename), self.mat.detach().cpu().numpy())
        savename = 'kernel_' + savename_suffix
        np.save(os.path.join(save_dir, savename), self.proj.weight.detach().cpu().numpy())
