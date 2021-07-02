"""Matrix multiply encoders

These encoders perform a matrix multiply on the vectorized image.
The same matrix is multiplied by each RGB channel separately to give R, G, and B vectors
For historical reasons, these encoders are referred to as 'Linear'

TODO:
    A lot repeated functionality for each (especially in initialization), perhaps write a base matmul class?
    Since implementing user_parameters, these have not been tested (because not in priority)
"""

import math
import torch
import torch.nn as nn
import numpy as np
import os
from torch.nn import functional as F
from models.encoders.util import scale_max_to_1, quantize
from models.encoders.base_encoder import BaseEncoder


# Linear only, returns three channel column vectors
class LinearOnly(BaseEncoder):
    """Encoder that applies a matrix multiply of proj_mat by the vectorized image

    User parameters:
        proj_mat_path (str): Numpy array containing proj_mat
        noise_std (float): Standard deviation of Gaussian noise that will be added to the output
        learn_noise (bool): Set to True to allow learning of the noise standard deviation
        with_transpose (bool): Set to True to perform proj_mat's transpose right after proj_mat
    """
    def __init__(self, opt, learn_noise=False, noise_std=0, proj_mat_path=None, with_transpose=False):
        super(LinearOnly, self).__init__()
        self.learn_noise = learn_noise
        self.noise_std = noise_std
        self.with_transpose = with_transpose
        if proj_mat_path:
            proj_mat = np.load(os.path.join(opt.root_data_dir, proj_mat_path))
        else:
            imdim = opt.input_dim_recog ** 2
            projdim = round(opt.meaurement_rate * imdim)
            # Defaults:
            # norm_std chosen such that outputs are approx. N(0,1) (see playground.py and black notebook for details)
            norm_std = 1 / ((imdim + 1) * projdim * 0.365) ** 0.25
            proj_mat = np.random.normal(0, norm_std, (projdim, imdim))  # random Gaussian matrix
        # Convert to Pytorch objects
        self.proj = nn.Linear(imdim, projdim, bias=False)
        self.proj.weight = nn.Parameter(torch.Tensor(proj_mat))
        self.noise = nn.Conv1d(1, 1, 1, bias=False)
        self.noise.weight = nn.Parameter(torch.Tensor([[[self.noise_std]]]))
        self.noise.requires_grad = True if self.learn_noise else False

    def forward(self, x):
        if x.size(1) == 1:  # grayscale
            xsize = x.size()
            x = x.view(xsize[0], -1)
            x = self.proj(x)
            x = x.unsqueeze(1)
            x = x.add(self.noise(torch.randn_like(x)))
        elif x.size(1) == 3:  # RGB
            # Red
            tmpR = x[:, 0, :, :].view(x.size(0), -1)
            tmpR = self.proj(tmpR)
            tmpR = tmpR.unsqueeze(1)
            tmpR = tmpR.add(self.noise(torch.randn_like(tmpR)))
            # Green
            tmpG = x[:, 1, :, :].view(x.size(0), -1)
            tmpG = self.proj(tmpG)
            tmpG = tmpG.unsqueeze(1)
            tmpG = tmpG.add(self.noise(torch.randn_like(tmpG)))
            # Blue
            tmpB = x[:, 2, :, :].view(x.size(0), -1)
            tmpB = self.proj(tmpB)
            tmpB = tmpB.unsqueeze(1)
            tmpB = tmpB.add(self.noise(torch.randn_like(tmpB)))
            x = torch.cat((tmpR, tmpG, tmpB), dim=1)
        return x

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        if not self.learn_noise:
            self.noise.requires_grad = False


# Performs a linear operation and then its transpose
# The transpose is disconnected from the linear layer, allowing it to be learned
class LinearWithFlexibleTranspose(BaseEncoder):
    """Encoder that applies a matrix multiply of proj_mat by the vectorized image followed by a backprojection

    The difference between this and LinearWithForcedTranspose is that for the latter, the backprojection
    is forced to be the transpose of the matrix multiply. Here, the backprojection has its own parameters and
    allows them to either be fixed or learnable.

        args:
            user_parameters
            opt
            proj_type (int): 0 is just a pass-through, 1 leaves backprojection fixed, 2 allows it to be learned

        User parameters:
            proj_mat_path (str): Numpy array containing proj_mat
            noise_std (float): Standard deviation of Gaussian noise that will be added to the output
            learn_noise (bool): Set to True to allow learning of the noise standard deviation
            quantize_bits (int): Number of bits for quantization (set to 0 for no quantization)
            random_backproject (bool): If true, backprojection is random Gaussian. If false, it is proj_mat's transpose.
    """

    def __init__(self, opt, proj_type=0, learn_noise=False, noise_std=0, proj_mat_path=None, quantize_bits=0,
                 random_backproject=False):

        super(LinearWithFlexibleTranspose, self).__init__()
        self.proj_type = proj_type
        imdim = opt.input_dim_recog ** 2
        projdim = round(opt.meaurement_rate * imdim)
        # Defaults:
        # norm_std chosen such that outputs are approx. N(0,1) (see playground.py and black notebook for details)
        norm_std = 1 / ((imdim + 1) * projdim * 0.365) ** 0.25
        proj_mat = np.random.normal(0, norm_std, (projdim, imdim))  # random Gaussian matrix
        self.learn_noise = False
        self.noise_std = 0
        self.quantize_bits = 0
        self.random_backproject = False
        # User input:
        if 'learn_noise' in user_parameters:
            self.learn_noise = user_parameters['learn_noise']
        if 'noise_std' in user_parameters:
            self.noise_std = user_parameters['noise_std']
        if 'proj_mat_path' in user_parameters:
            proj_mat = np.load(user_parameters['proj_mat_path'])
        if 'quantize_bits' in user_parameters:
            self.quantize_bits = user_parameters['quantize_bits']
            self.random_backproject = True
        if 'random_backproject' in user_parameters:
            self.random_backproject = user_parameters['random_backproject']
        # Create Pytorch objects
        proj_mat = torch.Tensor(proj_mat)
        if self.proj_type == 1 or self.proj_type == 2:
            proj_matT = proj_mat.transpose(1, 0)
            if self.random_backproject:
                nn.init.normal_(proj_matT)
            self.proj = nn.Linear(imdim, projdim, bias=False)
            self.projT = nn.Linear(projdim, imdim, bias=False)
            self.proj.weight = nn.Parameter(proj_mat)
            self.projT.weight = nn.Parameter(proj_matT)
            self.proj.weight.requires_grad = False  # fix these matrices
            if self.proj_type == 1:
                self.projT.weight.requires_grad = False
            else:
                self.projT.weight.requires_grad = True  # allow projT to be learned
        self.noise = nn.Parameter(torch.Tensor([[[self.noise_std]]]))
        self.noise.requires_grad = True if self.learn_noise else False

    def forward(self, x):
        if self.proj_type != 0:
            if x.size(1) == 1:  # grayscale
                xsize = x.size()
                x = x.view(xsize[0], -1)
                x = self.proj(x)
                x = self.projT(x)
                x = x.view(xsize[0], 1, xsize[2], xsize[3])
            elif x.size(1) == 3:  # RGB
                # Red
                tmpR = x[:, 0, :, :].view(x.size(0), -1)
                tmpR = self.proj(tmpR)
                tmpR = self.projT(tmpR)
                tmpR = tmpR.view(x.size(0), 1, x.size(2), x.size(3))
                # Green
                tmpG = x[:, 1, :, :].view(x.size(0), -1)
                tmpG = self.proj(tmpG)
                tmpG = self.projT(tmpG)
                tmpG = tmpG.view(x.size(0), 1, x.size(2), x.size(3))
                # Blue
                tmpB = x[:, 2, :, :].view(x.size(0), -1)
                tmpB = self.proj(tmpB)
                tmpB = self.projT(tmpB)
                tmpB = tmpB.view(x.size(0), 1, x.size(2), x.size(3))
                x = torch.cat((tmpR, tmpG, tmpB), dim=1)
                x = scale_max_to_1(x, clip_zero=False, with_negative=True)
                x = x.add(torch.mul(torch.randn_like(x), self.noise))
                if self.quantize_bits:
                    x = scale_max_to_1(x, clip_zero=False, with_negative=True)
                    x = quantize(x, num_bits=self.quantize_bits, with_negative=True)
        return x

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        if type(self.proj).__name__ == 'LinearWithFlexibleTranspose':
            if not self.proj.learn_noise:
                self.proj.noise.requires_grad = False
            if self.proj.proj_type == 1:  # do not learn transpose
                self.proj.projT.weight.requires_grad = False


def LinearWithFixedFlexibleTranspose(user_parameters, opt):
    """Returns an encoder that performs a matrix multiply followed by a fixed backprojection

        Args:
            user_parameters (dict): Dictionary of user parameters allowing the following keys:
                proj_mat_path (str): Numpy array containing proj_mat
                noise_std (float): Standard deviation of Gaussian noise that will be added to the output
                learn_noise (bool): Set to True to allow learning of the noise standard deviation
                quantize_bits (int): Number of bits for quantization (set to 0 for no quantization)
                random_backproject (bool): If true, backprojection is random Gaussian. Else, it is proj_mat's transpose.
            opt (parser)

        Yields:
            LinearWithFlexibleTranspose: the encoder object
    """
    return LinearWithFlexibleTranspose(user_parameters, opt, proj_type=1)


def LinearWithLearnedTranspose(user_parameters, opt):
    """Returns an encoder that performs a matrix multiply followed by a backprojection that is learnable

        Args:
            user_parameters (dict): Dictionary of user parameters allowing the following keys:
                proj_mat_path (str): Numpy array containing proj_mat
                noise_std (float): Standard deviation of Gaussian noise that will be added to the output
                learn_noise (bool): Set to True to allow learning of the noise standard deviation
                quantize_bits (int): Number of bits for quantization (set to 0 for no quantization)
                random_backproject (bool): If true, backprojection is random Gaussian. Else, it is proj_mat's transpose.
            opt (parser)

        Yields:
            LinearWithFlexibleTranspose: the encoder object
    """
    return LinearWithFlexibleTranspose(user_parameters, opt, proj_type=2)


# A network that performs a linear operation and then its transpose.
# The transpose is forced to be transpose of linear operator -- they are trained together
class LinearWithForcedTranspose(BaseEncoder):
    """Encoder that performs matrix multiply and the transpose of that matrix multiply

    User parameters:
        proj_mat_path (str): Numpy array containing proj_mat
        noise_std (float): Standard deviation of Gaussian noise that will be added to the output
        learn_noise (bool): Set to True to allow learning of the noise standard deviation
        with_transpose (bool): Set to True to perform proj_mat's transpose right after proj_mat

    TODO:
        merge with LinearOnly, check that freezing and unfreezing works
    """

    def __init__(self, user_parameters, opt):

        super(LinearWithForcedTranspose, self).__init__()
        imdim = opt.input_dim_recog ** 2
        projdim = round(opt.meaurement_rate * imdim)
        # Defaults:
        # norm_std chosen such that outputs are approx. N(0,1) (see playground.py and black notebook for details)
        norm_std = 1 / ((imdim + 1) * projdim * 0.365) ** 0.25
        proj_mat = np.random.normal(0, norm_std, (projdim, imdim))  # random Gaussian matrix
        self.learn_noise = False
        self.noise_std = 0
        # User input:
        if 'learn_noise' in user_parameters:
            self.learn_noise = user_parameters['learn_noise']
        if 'noise_std' in user_parameters:
            self.noise_std = user_parameters['noise_std']
        if 'proj_mat_path' in user_parameters:
            proj_mat = np.load(user_parameters['proj_mat_path'])

        # one layer that performs linear projection then its transpose
        self.proj = LinearWithTransposeLayer(imdim, projdim, init_weight=proj_mat, noise_std=self.noise_std,
                                             learn_noise=self.learn_noise)

    # pass x through the one layer with proper resizing
    def forward(self, x):
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
            tmpB = self.proj(tmpB)
            tmpB = tmpB.view(x.size(0), 1, x.size(2), x.size(3))
            x = torch.cat((tmpR, tmpG, tmpB), dim=1)
        return x

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        if not self.learn_noise:
            self.proj.noise.requires_grad = False


# a fully connected layer that would also perform its transpose
class LinearWithTransposeLayer(BaseEncoder):
    """A fully connected layer that would also perform its transpose -- used by LinearWithForcedTranspose

    TODO: make this private?
    """

    def __init__(self, in_features, out_features, bias=False, init_weight=None, noise_std=0, learn_noise=False):
        super(LinearWithTransposeLayer, self).__init__()
        if init_weight is not None:
            assert init_weight.shape[0] == out_features and init_weight.shape[1] == in_features, \
                "Provided initial weights do not match provided dimensions"
        self.in_features = in_features
        self.out_features = out_features
        self.init_weight = init_weight
        self.noise_std = noise_std
        self.has_bias = bias
        self.learn_noise = learn_noise
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias_proj = nn.Parameter(torch.Tensor(out_features))
            self.bias_t = nn.Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias_proj', None)
            self.register_parameter('bias_t', None)
        # Noise
        self.noise = nn.Parameter(torch.Tensor([[[self.noise_std]]]))
        self.noise.requires_grad = True if self.learn_noise else False
        # Initialize weight
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weight is not None:
            self.weight = nn.Parameter(torch.Tensor(self.init_weight))
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.has_bias:
            fan_in_proj, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound_proj = 1 / math.sqrt(fan_in_proj)
            fan_in_t, _ = nn.init._calculate_fan_in_and_fan_out(self.weight.t())
            bound_t = 1 / math.sqrt(fan_in_proj)
            nn.init.uniform_(self.bias_proj, -bound_proj, bound_proj)
            nn.init.uniform_(self.bias_t, -bound_t, bound_t)

    def forward(self, input):
        # project
        x = F.linear(input, self.weight, self.bias_proj)
        # add noise
        x = x.unsqueeze(1)
        x = x.add(F.conv1d(torch.randn_like(x), self.noise))
        x = x.squeeze()
        # transpose
        return F.linear(x, self.weight.t(), self.bias_t)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias_proj={}, bias_t={}'.format(
            self.in_features, self.out_features, self.bias_proj, self.bias_t is not None
        )
