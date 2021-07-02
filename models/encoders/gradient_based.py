"""Gradient Based Encoders

Encoders that are based on gradient-based or edge-based operations.
Examples include Gabor filters, edge extractors, oriented gradients, etc.
"""

import skimage
from skimage.feature import local_binary_pattern, hog
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.encoders.util import normalize, scale_max_to_1, quantize
from pytorch_sift import SIFTNet
from models.encoders.base_encoder import BaseEncoder


class GaborConv2d(BaseEncoder):
    """

    User parameters:
        device (int): GPU index
        kernel_size (int)
        in_channels (int)
        out_channels (int)
        avg (bool)
        learn_noise (bool)
        mxp (bool)
        quantize_bits (int)
        quantize (bool)
    """

    def __init__(self, opt, kernel_size=11, in_channels=1, out_channels=4, avg=True, learn_noise=True,
                 mxp=False, quantize_bits=4, quantize=True):
        super(GaborConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.x0 = torch.ceil(torch.Tensor([self.kernel_size / 2]))[0]
        self.y0 = torch.ceil(torch.Tensor([self.kernel_size / 2]))[0]
        self.device = opt.gpu_idx
        self.avgpool = nn.AvgPool2d(16, 16)
        self.mxpool = nn.MaxPool2d(16,16)
        self.avg = avg
        self.mxp = mxp
        self.learn_noise = learn_noise
        self.noise = nn.Parameter(torch.Tensor([0])).type(torch.Tensor)
        self.noise.requires_grad = True if self.learn_noise else False
        self.quantize = quantize
        self.bits = quantize_bits

        # Pytorch objects
        self.freq = nn.Parameter(torch.Tensor([[0.6], [0.6], [0.6], [0.6]])).type(torch.Tensor)
        self.theta = nn.Parameter(torch.Tensor([[0], [3.14 / 4], [3.14 / 2], [3.14]]).type(torch.Tensor))

    def forward(self, input):
        y, x = torch.meshgrid([torch.linspace(-self.x0 + 1, self.x0, self.kernel_size),
                               torch.linspace(-self.y0 + 1, self.y0, self.kernel_size)])
        x = x.to(self.device)
        y = y.to(self.device)
        weight = torch.empty((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size),
                             requires_grad=False).to(self.device)
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                theta = self.theta[i, j].expand_as(y)
                rotx = x * torch.cos(theta) + y * torch.sin(theta)
                roty = -x * torch.sin(theta) + y * torch.cos(theta)
                g = torch.zeros(y.shape)
                g = torch.exp(-0.5 * ((rotx ** 2 + roty ** 2) / (4 + 1e-3) ** 2))
                g = g * torch.cos(self.freq[i,j] * rotx)
                weight[i, j, :, :] = g

        tmpR = input[:, 0, :, :].unsqueeze(1)
        # tmpRX = F.conv2d(tmpR, proj_matX, None, 1, 1)
        # tmpRY = F.conv2d(tmpR, proj_matY, None, 1, 1)
        # tmpR = torch.sqrt((tmpRX ** 2) + (tmpRY ** 2))
        tmpR = F.conv2d(tmpR, weight, None, 1, 5)
        # tmpR = tmpR.add(self.noise(torch.randn_like(tmpR)))
        # Green
        tmpG = input[:, 1, :, :].unsqueeze(1)
        # tmpGX = F.conv2d(tmpG, proj_matX, None, 1, 1)
        # tmpGY = F.conv2d(tmpG, proj_matY, None, 1, 1)
        # tmpG = torch.sqrt((tmpGX ** 2) + (tmpGY ** 2))
        tmpG = F.conv2d(tmpG, weight, None, 1, 5)
        # tmpG = tmpG.add(self.noise(torch.randn_like(tmpG)))
        # Blue
        tmpB = input[:, 2, :, :].unsqueeze(1)
        # tmpBX = F.conv2d(tmpB, proj_matX, None, 1, 1)
        # tmpBY = F.conv2d(tmpB, proj_matY, None, 1, 1)
        # tmpB = torch.sqrt((tmpBX ** 2) + (tmpBY ** 2))
        tmpB = F.conv2d(tmpB, weight, None, 1, 5)
        # tmpB = tmpB.add(self.noise(torch.randn_like(tmpB)))
        op = torch.cat((tmpR, tmpG, tmpB), dim=1)
        if self.mxp:
            # print('Max Pooling')
            op = self.mxpool(op)
        op = op.add(torch.mul(torch.randn_like(op), self.noise))
        if self.quantize:
            # print(self.quantize)
            qmin = 0.
            qmax = 2. ** self.bits - 1.
            min_value = op.min()
            max_value = op.max()
            scale = (max_value - min_value) / (qmax - qmin)
            scale = max(scale, 1e-8)
            op = op.add(-min_value).div(scale).add(qmin)

            ### quantize
            # x.clamp_(qmin, qmax).round_()  # quantize
            op = op.clamp(qmin, qmax)
            y = op.detach() # detech to round and prop gradient through round
            y = y - y.round()
            op = op - y

            op = op.add(-qmin).mul(scale).add(min_value)  # dequantize
        op = F.interpolate(op, size=(64,64))
        print(op.shape)
        return op


class OrientedGradientsNew(BaseEncoder):

    def __init__(self, opt, kernel_size=11, in_channels=1, out_channels=4, avg=True, learn_noise=True,
                 quantize_bits=4, quantize=True):
        super(OrientedGradientsNew, self).__init__()
        self.freq = nn.Parameter(torch.Tensor([0.6])).type(torch.Tensor)
        self.theta = nn.Parameter(torch.tensor([[0], [3.14 / 4], [3.14 / 2], [3.14]]).type(torch.Tensor))
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.x0 = torch.ceil(torch.Tensor([self.kernel_size / 2]))[0]
        self.y0 = torch.ceil(torch.Tensor([self.kernel_size / 2]))[0]
        self.device = opt.gpu_idx
        self.avgpool = nn.AvgPool2d(16, 16)
        self.avg = avg
        self.learn_noise = learn_noise
        self.noise = nn.Parameter(torch.Tensor([0])).type(torch.Tensor)
        self.noise.requires_grad = True if self.learn_noise else False
        self.quantize = quantize
        self.bits = quantize_bits

    def forward(self, input):
        y, x = torch.meshgrid([torch.linspace(-self.x0 + 1, self.x0, self.kernel_size),
                               torch.linspace(-self.y0 + 1, self.y0, self.kernel_size)])
        x = x.to(self.device)
        y = y.to(self.device)
        proj_matX = torch.Tensor([[np.array([[-1, 0, 1], [-2, 0, +2], [-1, 0, 1]])]]).to(self.device)
        proj_matY = torch.Tensor([[np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])]]).to(self.device)
        weight = torch.empty((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size),
                             requires_grad=False).to(self.device)
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                theta = self.theta[i, j].expand_as(y)
                rotx = x * torch.cos(theta) + y * torch.sin(theta)
                roty = -x * torch.sin(theta) + y * torch.cos(theta)
                g = torch.zeros(y.shape)
                g = torch.exp(-0.5 * ((rotx ** 2 + roty ** 2) / (4 + 1e-3) ** 2))
                g = g * torch.cos(self.freq * rotx)
                weight[i, j, :, :] = g

        tmpR = input[:, 0, :, :].unsqueeze(1)
        tmpRX = F.conv2d(tmpR, proj_matX, None, 1, 1)
        tmpRY = F.conv2d(tmpR, proj_matY, None, 1, 1)
        tmpR = torch.sqrt((tmpRX ** 2) + (tmpRY ** 2))
        tmpR = F.conv2d(tmpR, weight, None, 1, 5)
        # tmpR = tmpR.add(self.noise(torch.randn_like(tmpR)))
        # Green
        tmpG = input[:, 1, :, :].unsqueeze(1)
        tmpGX = F.conv2d(tmpG, proj_matX, None, 1, 1)
        tmpGY = F.conv2d(tmpG, proj_matY, None, 1, 1)
        tmpG = torch.sqrt((tmpGX ** 2) + (tmpGY ** 2))
        tmpG = F.conv2d(tmpG, weight, None, 1, 5)
        # tmpG = tmpG.add(self.noise(torch.randn_like(tmpG)))
        # Blue
        tmpB = input[:, 2, :, :].unsqueeze(1)
        tmpBX = F.conv2d(tmpB, proj_matX, None, 1, 1)
        tmpBY = F.conv2d(tmpB, proj_matY, None, 1, 1)
        tmpB = torch.sqrt((tmpBX ** 2) + (tmpBY ** 2))
        tmpB = F.conv2d(tmpB, weight, None, 1, 5)
        # tmpB = tmpB.add(self.noise(torch.randn_like(tmpB)))
        op = torch.cat((tmpR, tmpG, tmpB), dim=1)
        if self.avg:
            op = self.avgpool(op)
            op = F.interpolate(op, size=(64, 64))
        op = op.add(torch.mul(torch.randn_like(op), self.noise))
        if self.quantize:
            # print(self.quantize)
            qmin = 0.
            qmax = 2. ** self.bits - 1.
            min_value = op.min()
            max_value = op.max()
            scale = (max_value - min_value) / (qmax - qmin)
            scale = max(scale, 1e-8)
            op = op.add(-min_value).div(scale).add(qmin)

            ### quantize
            #        import pdb; pdb.set_trace()
            # x.clamp_(qmin, qmax).round_()  # quantize
            op = op.clamp(qmin, qmax)
            y = op.detach()  # detech to round and prop gradient through round
            y = y - y.round()
            op = op - y

            op = op.add(-qmin).mul(scale).add(min_value)  # dequantize
        return op


class OrientedGradientsCCMax(BaseEncoder):

    def __init__(self, opt, kernel_size=11, in_channels=1, out_channels=4, avg=True, learn_noise=False):
        super(OrientedGradientsCCMax, self).__init__()
        self.freq = nn.Parameter(torch.Tensor([0.6])).type(torch.Tensor)
        self.theta = nn.Parameter(torch.tensor([[0], [3.14 / 4], [3.14 / 2], [3.14]]).type(torch.Tensor))
        self.kernel_size = kernel_size
        self.x0 = torch.ceil(torch.Tensor([self.kernel_size / 2]))[0]
        self.y0 = torch.ceil(torch.Tensor([self.kernel_size / 2]))[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.x0 = torch.ceil(torch.Tensor([self.kernel_size / 2]))[0]
        self.y0 = torch.ceil(torch.Tensor([self.kernel_size / 2]))[0]
        self.device = opt.gpu_idx
        self.avgpool = nn.AvgPool2d(16, 16)
        self.avg = avg
        self.learn_noise = learn_noise
        self.noise = nn.Parameter(torch.Tensor([0])).type(torch.Tensor)
        self.noise.requires_grad = True if self.learn_noise else False

    def forward(self, input):
        y, x = torch.meshgrid([torch.linspace(-self.x0 + 1, self.x0, self.kernel_size),
                               torch.linspace(-self.y0 + 1, self.y0, self.kernel_size)])
        x = x.to(self.device)
        y = y.to(self.device)
        proj_matX = torch.Tensor([[np.array([[-1, 0, 1], [-2, 0, +2], [-1, 0, 1]])]]).to(self.device)
        proj_matY = torch.Tensor([[np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])]]).to(self.device)
        weight = torch.empty((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size),
                             requires_grad=False).to(self.device)
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                theta = self.theta[i, j].expand_as(y)
                rotx = x * torch.cos(theta) + y * torch.sin(theta)
                roty = -x * torch.sin(theta) + y * torch.cos(theta)
                g = torch.zeros(y.shape)
                g = torch.exp(-0.5 * ((rotx ** 2 + roty ** 2) / (4 + 1e-3) ** 2))
                g = g * torch.cos(self.freq * rotx)
                weight[i, j, :, :] = g

        tmpR = input[:, 0, :, :].unsqueeze(1)
        tmpRX = F.conv2d(tmpR, proj_matX, None, 1, 1)
        tmpRY = F.conv2d(tmpR, proj_matY, None, 1, 1)
        tmpR = torch.sqrt((tmpRX ** 2) + (tmpRY ** 2))

        # tmpR = tmpR.add(self.noise(torch.randn_like(tmpR)))
        # Green
        tmpG = input[:, 1, :, :].unsqueeze(1)
        tmpGX = F.conv2d(tmpG, proj_matX, None, 1, 1)
        tmpGY = F.conv2d(tmpG, proj_matY, None, 1, 1)
        tmpG = torch.sqrt((tmpGX ** 2) + (tmpGY ** 2))
        # tmpG = tmpG.add(self.noise(torch.randn_like(tmpG)))
        # Blue
        tmpB = input[:, 2, :, :].unsqueeze(1)
        tmpBX = F.conv2d(tmpB, proj_matX, None, 1, 1)
        tmpBY = F.conv2d(tmpB, proj_matY, None, 1, 1)
        tmpB = torch.sqrt((tmpBX ** 2) + (tmpBY ** 2))
        # tmpB = tmpB.add(self.noise(torch.randn_like(tmpB)))
        op = torch.cat((tmpR, tmpG, tmpB), dim=1)
        op = torch.max(op, 1)[0].unsqueeze(1)
        op = F.conv2d(op, weight, None, 1, 5)
        if self.avg:
            op = self.avgpool(op)
            op = F.interpolate(op, size=(64, 64))
        op = op.add(torch.mul(torch.randn_like(op), self.noise))
        return op


class Gabor(BaseEncoder):

    def __init__(self, frequency=1, theta=0, n_stds=5,input_dim=64, quantize_bits=16, noise_std=0, normalize=True):

        super(Gabor, self).__init__()
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
        #gab = np.load('gab_0p24_15sd.npy')
        gab=sio.loadmat('gab4.mat')['gab4']

        self.proj_mat = gab

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


class GaborBank(BaseEncoder):

    def __init__(self, frequency=1, theta=0, n_stds=5,input_dim=64, quantize_bits=0, noise_std=0, normalize=True):

        super(GaborBank, self).__init__()
        # self.frequency = frequency
        # self.theta = theta
        # self.n_stds = n_stds
        self.quantize_bits = quantize_bits
        self.noise_std = noise_std if noise_std is not None else 0
        self.normalize = normalize

        self.proj = nn.Conv2d(1, 32, input_dim + 1, padding=int(input_dim / 2), bias=False)
        self.proj.requires_grad = False
        # Gaussian blur kernel using numpy
        # dirac = np.zeros((input_dim + 1, input_dim + 1))
        # dirac[int(input_dim / 2), int(input_dim / 2)] = 1
        #gab = sf.gabor_kernel(frequency=self.frequency, theta=self.theta, n_stds=self.n_stds).real
        #gab = np.load('gab_0p24_15sd.npy')
        gab=sio.loadmat('gabor_2:4_0:135.mat')['h']

        self.proj_mat = gab

        self.proj.weight = nn.Parameter(torch.Tensor(gab), requires_grad=False)

        # noise layer
        self.noise = nn.Conv2d(1, 1, 1, bias=False)
        self.noise.weight = nn.Parameter(torch.Tensor([[[[self.noise_std]]]]))

    def forward(self, x):
        if self.normalize:
            if x.size(1) == 1:
                x = normalize(x, mean=0.485, std=0.229)
            elif x.size(1) == 3:
                x = normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


        if x.size(1) == 1:  # grayscale
            x = self.proj(x)
            x = x.add(self.noise(torch.rand_like(x)))
        elif x.size(1) == 3:  # RGB
            # Red
            tmpR = x[:, 0, :, :].unsqueeze(1)
            tmpR = self.proj(tmpR)
            #tmpR = tmpR.add(self.noise(torch.randn_like(tmpR)))
            # Green
            tmpG = x[:, 1, :, :].unsqueeze(1)
            tmpG = self.proj(tmpG)
            #tmpG = tmpG.add(self.noise(torch.randn_like(tmpG)))
            # Blue
            tmpB = x[:, 2, :, :].unsqueeze(1)
            tmpB = self.proj(tmpB)
            #tmpB = tmpB.add(self.noise(torch.randn_like(tmpB)))
            x = torch.cat((tmpR, tmpG, tmpB), dim=1)
        x = scale_max_to_1(x,with_negative=True)
        x = quantize(x, num_bits=self.quantize_bits)
        if self.normalize:
            if x.size(1) == 1:
                x = normalize(x, mean=1.1723, std=5.5470)
            elif x.size(1) == 3:
                x = normalize(x, mean=[1.1723, 1.1022, 0.9813], std=[5.5470, 5.4861, 5.4983])
        return x


class Canny(BaseEncoder):

    def __init__(self,input_dim=64, quantize_bits=0, noise_std=0, normalize=True,gpu_idx=0, canny_sigma=3):

        super(Canny, self).__init__()
        self.gpu_idx=gpu_idx
        self.quantize_bits=quantize_bits
        self.sigma=canny_sigma

    def forward(self, x):

        x=x.cpu().detach().numpy()
        y=np.zeros((x.shape[0],x.shape[2],x.shape[3]))
        for i in range(x.shape[0]):
            #print('Calculating for:'+str(i)+'of:'+str(x.shape[0]))
            y[i,:,:]=skimage.feature.canny(x[i,0,:,:],sigma=self.sigma)
        y=y.astype('float32')
        y=torch.from_numpy(y).unsqueeze(1).cuda(self.gpu_idx)
        y = quantize(y, num_bits=self.quantize_bits)
        return y


class Edge(BaseEncoder):

    def __init__(self,input_dim=64, quantize_bits=2, noise_std=0, normalize=True,gpu_idx=0, edge_sigma=4, edge_threshold=0.035):

        super(Edge, self).__init__()
        self.gpu_idx=gpu_idx
        self.quantize_bits=quantize_bits
        self.sigma=edge_sigma
        self.threshold=edge_threshold

    def forward(self, x):

        x=x.cpu().detach().numpy()
        y=np.zeros((x.shape[0],x.shape[2],x.shape[3]))
        for i in range(x.shape[0]):
            #print('Calculating for:'+str(i)+'of:'+str(x.shape[0]))
            y[i,:,:]=np.greater(skimage.filters.sobel(skimage.filters.gaussian(x[i,0,:,:],sigma=self.sigma)),self.threshold)

        y=y.astype('float32')
        y=torch.from_numpy(y).unsqueeze(1).cuda(self.gpu_idx)
        y = quantize(y, num_bits=self.quantize_bits)
        return y


class Gradient(BaseEncoder):

    def __init__(self,input_dim=64, quantize_bits=2, noise_std=0, normalize=False,gpu_idx=0):

        super(Gradient, self).__init__()
        self.gpu_idx=gpu_idx
        self.quantize_bits=quantize_bits
        self.normalize=normalize

    def forward(self, x):

        x=x.cpu().detach().numpy()
        y=np.zeros((x.shape[0],x.shape[2],x.shape[3]))
        for i in range(x.shape[0]):
            #print('Calculating for:'+str(i)+'of:'+str(x.shape[0]))
            y[i,:,:]=skimage.filters.sobel(x[i,0,:,:])

        y=y.astype('float32')
        y=torch.from_numpy(y).unsqueeze(1).cuda(self.gpu_idx)
        y = scale_max_to_1(y)
        y = quantize(y, num_bits=self.quantize_bits)
        # if self.normalize:
        #     if y.size(1) == 1:
        #         y = normalize(y, mean=0.485, std=0.229)
        #     elif y.size(1) == 3:
        #         y = normalize(y, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return y


class HOG(BaseEncoder):

    def __init__(self,input_dim=64, quantize_bits=0, gpu_idx=0, dense=False):

        super(HOG, self).__init__()
        self.gpu_idx = gpu_idx
        self.quantize_bits = quantize_bits
        self.dense = dense

    def forward(self, x):

        x = x.cpu().detach().numpy()
        max_patchsize = 32
        imp = np.zeros((x.shape[2]+max_patchsize-1, x.shape[3]+max_patchsize-1, x.shape[1]))
        pad_top = int(max_patchsize/2) - 1
        pad_bottom = int(max_patchsize/2)
        pad_left = int(max_patchsize/2) - 1
        pad_right = int(max_patchsize/2)
        orientations = 8
        num_scale = 3
        y = np.zeros((x.shape[0], int(orientations*num_scale), x.shape[2], x.shape[3]))
        cpb = 1
        f_map = np.zeros((int(orientations*num_scale), x.shape[2], x.shape[3]))
        psizes = [8, 16, 32]
        for p in range(x.shape[0]):
            print(str(p+1)+' of '+str(x.shape[0]))
            for ch in range(x.shape[1]):
                imp[:, :, ch] = np.pad(x[p, ch, :, :], ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant')
            if self.dense:
                for i in range(pad_top, pad_top+x.shape[2]):
                    for j in range(pad_left, pad_left+x.shape[3]):
                        ps = 8
                        patch1 = imp[i-int(ps/2)+1:i+int(ps/2)+1, j-int(ps/2)+1:j+int(ps/2)+1, :]
                        ps = 16
                        patch2 = imp[i-int(ps/2)+1:i+int(ps/2)+1, j-int(ps/2)+1:j+int(ps/2)+1, :]
                        ps = 32
                        patch3 = imp[i-int(ps/2)+1:i+int(ps/2)+1, j-int(ps/2)+1:j+int(ps/2)+1, :]
                        f_map[0:orientations, i-pad_top, j-pad_left] = hog(patch1, orientations=orientations, pixels_per_cell=(patch1.shape[0], patch1.shape[1]),
                                    cells_per_block=(cpb, cpb), visualize=False, multichannel=True)
                        f_map[orientations:2*orientations, i-pad_top, j-pad_left] = hog(patch2, orientations=orientations, pixels_per_cell=(patch2.shape[0], patch2.shape[1]),
                                    cells_per_block=(cpb, cpb), visualize=False, multichannel=True)
                        f_map[2*orientations:3*orientations, i-pad_top, j-pad_left] = hog(patch3, orientations=orientations, pixels_per_cell=(patch3.shape[0], patch3.shape[1]),
                                    cells_per_block=(cpb, cpb), visualize=False, multichannel=True)

            else:
                for h in range(len(psizes)):
                    ps = psizes[h]
                    h = 1
                    fmap = np.zeros((int(x.shape[2]/ps), int(x.shape[3]/ps), orientations))
                    for i in range(pad_top, pad_top+x.shape[2], ps):
                        k = 1
                        for j in range(pad_left, pad_left+x.shape[3], ps):
                            patch = imp[i-int(ps/2)+1:i+int(ps/2)+1, j-int(ps/2)+1:j+int(ps/2)+1, :]
                            fmap[h-1, k-1, 0:8] = hog(patch, orientations=orientations, pixels_per_cell=(patch.shape[0], patch.shape[1]),
                                        cells_per_block=(cpb, cpb), visualize=False, multichannel=True)
                            k += 1
                        h += 1
                    f_map[int(h*orientations):int((h+1)*orientations), :, :] = np.swapaxes(np.swapaxes(skimage.transform.resize(fmap, (x.shape[2], x.shape[3])), 0, 2), 1, 2)
            y[p, :, :, :] = f_map
        y = y.astype('float32')
        y = torch.from_numpy(y).cuda(self.gpu_idx)
        y = quantize(y, num_bits=self.quantize_bits)
        return y


class SIFT(BaseEncoder):

    def __init__(self,input_dim=64, quantize_bits=0, gpu_idx=0, dense=False):

        super(SIFT, self).__init__()
        self.gpu_idx = gpu_idx
        self.quantize_bits = quantize_bits
        self.dense = dense
        self.SIFTNet1 = SIFTNet(patch_size=8,num_spatial_bins = 1).cuda(self.gpu_idx)
        self.SIFTNet2 = SIFTNet(patch_size=16,num_spatial_bins = 1).cuda(self.gpu_idx)
        self.SIFTNet3 = SIFTNet(patch_size=32,num_spatial_bins = 1).cuda(self.gpu_idx)

    def forward(self, x):

        x = x.cpu().detach().numpy()
        max_patchsize = 32
        imp = np.zeros((x.shape[2]+max_patchsize-1, x.shape[3]+max_patchsize-1, x.shape[1]))
        pad_top = int(max_patchsize/2) - 1
        pad_bottom = int(max_patchsize/2)
        pad_left = int(max_patchsize/2) - 1
        pad_right = int(max_patchsize/2)
        orientations = 8
        num_scale = 3
        y = np.zeros((x.shape[0], int(orientations*num_scale), x.shape[2], x.shape[3]))
        cpb = 1
        f_map = torch.zeros((int(orientations*num_scale), x.shape[2], x.shape[3]))
        psizes = [8, 16, 32]
        for p in range(x.shape[0]):
            print(str(p+1)+' of '+str(x.shape[0]))
            for ch in range(x.shape[1]):
                imp[:, :, ch] = np.pad(x[p, ch, :, :], ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant')
            if self.dense:
                for i in range(pad_top, pad_top+x.shape[2]):
                    for j in range(pad_left, pad_left+x.shape[3]):
                        ps = 8
                        patch1 = imp[i-int(ps/2)+1:i+int(ps/2)+1, j-int(ps/2)+1:j+int(ps/2)+1, 1]
                        patch1 = patch1.astype('float32')
                        ps = 16
                        patch2 = imp[i-int(ps/2)+1:i+int(ps/2)+1, j-int(ps/2)+1:j+int(ps/2)+1, 1]
                        patch2 = patch2.astype('float32')
                        ps = 32
                        patch3 = imp[i-int(ps/2)+1:i+int(ps/2)+1, j-int(ps/2)+1:j+int(ps/2)+1, 1]
                        patch3 = patch3.astype('float32')
                        with torch.no_grad():
                            f_map[0:orientations, i-pad_top, j-pad_left] = self.SIFTNet1(torch.from_numpy(patch1).cuda(self.gpu_idx).unsqueeze(0).unsqueeze(0))
                            f_map[orientations:2*orientations, i-pad_top, j-pad_left] = self.SIFTNet2(torch.from_numpy(patch2).cuda(self.gpu_idx).unsqueeze(0).unsqueeze(0))
                            f_map[2*orientations:3*orientations, i-pad_top, j-pad_left] =self.SIFTNet3(torch.from_numpy(patch3).cuda(self.gpu_idx).unsqueeze(0).unsqueeze(0))

            else:
                for h in range(len(psizes)):
                    ps = psizes[h]
                    h = 1
                    fmap = torch.zeros((int(x.shape[2]/ps), int(x.shape[3]/ps), orientations))
                    for i in range(pad_top, pad_top+x.shape[2], ps):
                        k = 1
                        for j in range(pad_left, pad_left+x.shape[3], ps):
                            patch = imp[i-int(ps/2)+1:i+int(ps/2)+1, j-int(ps/2)+1:j+int(ps/2)+1, :]
                            fmap[h-1, k-1, 0:8] = hog(patch, orientations=orientations, pixels_per_cell=(patch.shape[0], patch.shape[1]),
                                        cells_per_block=(cpb, cpb), visualize=False, multichannel=True)
                            k += 1
                        h += 1
                    f_map[int(h*orientations):int((h+1)*orientations), :, :] = np.swapaxes(np.swapaxes(skimage.transform.resize(fmap, (x.shape[2], x.shape[3])), 0, 2), 1, 2)
            y[p, :, :, :] = f_map
        y = y.astype('float32')
        y = torch.from_numpy(y).cuda(self.gpu_idx)
        y = quantize(y, num_bits=self.quantize_bits)
        return y


class HOGNew(BaseEncoder):

    def __init__(self,input_dim=64, quantize_bits=0, gpu_idx=0, orientations=4, scales=(16,32), noise_std=0.01):

        super(HOGNew, self).__init__()
        self.gpu_idx = gpu_idx
        self.quantize_bits = quantize_bits
        self.orientations = orientations
        self.scales = scales
        self.num_scales = len(scales)
        self.noise_std = noise_std

    def forward(self, x):
        f_map = np.zeros((x.shape[2],x.shape[3],self.num_scales*self.orientations))
        x = x.cpu().detach().numpy()
        y = np.zeros((x.shape[0], int(self.orientations * self.num_scales), x.shape[2], x.shape[3]))
        for p in range(x.shape[0]):
            for ns in range(self.num_scales):
                fd = hog(np.swapaxes(np.swapaxes(x[p,:,:,:],0,2),0,1),orientations=self.orientations, pixels_per_cell=(self.scales[ns],self.scales[ns]), cells_per_block=(1,1), visualize=False, multichannel=True)
                f_map[:,:,int(ns*self.orientations):int((ns+1)*self.orientations)] = skimage.transform.resize(np.reshape(fd,[int(np.sqrt(fd.shape[0]/self.orientations)),int(np.sqrt(fd.shape[0]/self.orientations)),self.orientations]),(64,64))
            y[p, :, :, :] = np.swapaxes(np.swapaxes(f_map,0,2),1,2)
        y = y.astype('float32')
        y = torch.from_numpy(y).cuda(self.gpu_idx)
        noise_tensor = self.noise_std*torch.randn_like(y)
        y = y + noise_tensor
        y = quantize(y, num_bits=self.quantize_bits)
        return y