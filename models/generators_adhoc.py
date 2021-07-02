import torch.nn as nn
import torch
import numpy as np
import math
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter

raise NameError('generators_adhoc.py is deprecated. Pls use models.encoders instaad')

#
# def quantize(input, num_bits=16, with_negative=False):
#     # assumes input is from 0 to 1
#     if num_bits:
#         if with_negative:  # assumes input is from -1 to 1
#             input = (input + 1) / 2
#             output = (input * (2. ** num_bits - 1)).round() / (2. ** num_bits - 1)
#             output = (output * 2) - 1
#         else:
#             output = (input * (2.**num_bits - 1)).round() / (2.**num_bits - 1)
#     else:  # 0 bits means do not quantize
#         output = input
#     return output
#
#
# # assume image's minimum is 0, divide image by its maximum value (so that it maximum value is 1)
# # the maximum value is the maximum across all pixels in all channels for each image
# def scale_max_to_1(input, clip_zero=False, with_negative=False):
#     if clip_zero:
#         input = torch.clamp(input, min=0)
#     if with_negative:
#         max_per_im = input.view(input.size(0), -1).abs().max(dim=1)
#     else:
#         max_per_im = input.view(input.size(0), -1).max(dim=1)
#     max_per_im = max_per_im[0].view(-1, 1, 1, 1)
#     return input / max_per_im
#
#
# def normalize(input, mean=None, std=None):
#     if mean is None:
#         mean = [0, 0, 0]
#     if std is None:
#         std = [1, 1, 1]
#     mean = torch.Tensor(mean)
#     mean = mean.view(1, -1, 1, 1)
#     std = torch.Tensor(std)
#     std = std.view(1, -1, 1, 1)
#     if input.is_cuda:
#         mean = mean.cuda(input.get_device())
#         std = std.cuda(input.get_device())
#     return (input - mean) / std


