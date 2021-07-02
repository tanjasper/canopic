import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import *

# maximize the entropy (make it as close to uniform) -- equivalent to minimizing negative entropy
class NegativeEntropyLoss(nn.modules.loss._Loss):

    def __init__(self, topk=101, num_classes=None):
        super(NegativeEntropyLoss, self).__init__()
        self.topk = topk

    def forward(self, input, target):
        # solve for probabilities and log probs
        probs = F.softmax(input, dim=1)
        logprobs = F.log_softmax(input, dim=1)

        # take topk of them
        probs = probs.topk(self.topk)[0]
        logprobs = logprobs.topk(self.topk)[0] # should take the same topk since log is strictly increasing

        # entropy is the negative inner product of the two (not taking the negative gives us "negative entropy")
        loss = (probs * logprobs).sum(dim=1)

        # take mean across inputs in batch
        loss = loss.mean()
        return loss / self.topk


# Take the norm of all pairwise differences among probabilities for odd (even) digits if target is odd (even).
class FacesPairwiseNormLoss(nn.modules.loss._Loss):

    def __init__(self, norm=2, num_classes=100, topk=100, log=True):
        super(FacesPairwiseNormLoss, self).__init__()
        self.norm = norm
        self.num_classes = num_classes
        self.topk = topk
        self.log = log

    def forward(self, input, target):

        if self.log:
            softmax = nn.LogSoftmax(1)
        else:
            softmax = nn.Softmax(1)
        input = softmax(input)

        # take topk probabilities for each class
        input = input.topk(self.topk)[0]

        # generate all pairwise indices
        indices1 = []
        indices2 = []
        for i in range(self.topk-1):
            indices1 = indices1 + [i]*(self.topk-i-1)
            indices2 = indices2 + list(range(i+1, self.topk))

        # take the norm of all pairwise distances for each input
        pdist = nn.PairwiseDistance(p=self.norm)
        diffs = pdist(input[:, indices1], input[:, indices2])
        loss = diffs.mean()  # take mean across all inputs in batch
        return loss / self.topk


class EntropyLoss(nn.modules.loss._Loss):
    def __init__(self, topk=100, num_classes=100, size_average=None, reduce=None, reduction='mean'):
        super(EntropyLoss, self).__init__(size_average, reduce, reduction)

    # input is not probability distribution of output classes
    def forward(self, input):
        input = F.softmax(input, dim=1)
        if (input < 0).any() or (input > 1).any():
            raise Exception('Entropy Loss takes probabilities 0<=input<=1')

        input = input + 1e-16  # for numerical stability while taking log
        H = torch.mean(torch.sum(input * torch.log(input), dim=1))

        return -H


def FacesPairwiseL1NormLoss(num_classes, topk):

    return FacesPairwiseNormLoss(norm=1, num_classes=num_classes, topk=topk)


def FacesPairwiseL2NormLoss(num_classes, topk):

    return FacesPairwiseNormLoss(norm=2, num_classes=num_classes, topk=topk)


class NegativeCrossEntropyLoss(nn.modules.loss._WeightedLoss):

    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='elementwise_mean', num_classes=None, topk=None):
        super(NegativeCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return -F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)





