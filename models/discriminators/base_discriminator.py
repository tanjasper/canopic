import torch.nn as nn

class BaseDiscriminator(nn.Module):
    """Base class for discriminator classes
    Basic functions such as freeze() and unfreeze() are implemented here.
    The purpose of this class is to avoid rewriting common functions, however one is free to
    override these functions if needed.
    """

    def __init__(self):
        super(BaseDiscriminator, self).__init__()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True