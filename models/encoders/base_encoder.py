import torch.nn as nn

class BaseEncoder(nn.Module):
    """Base class for encoder classes
    Basic functions such as freeze() and unfreeze() are implemented here.
    The purpose of this class is to avoid rewriting common functions, however one is free to
    override these functions if needed.
    """

    def __init__(self, never_unfreeze=False, skip_init_loading=False):
        """

        Args:
            never_unfreeze (bool): if true, the parameters of this encoder will always have requires_grad=False
            skip_init (bool): if true, skip any file-loading involved in initialization (useful if the loaded
                file will simply be overwritten anyway, especially if file does not exist in running machine)
        """
        super(BaseEncoder, self).__init__()
        self.never_unfreeze = never_unfreeze
        self.skip_init_loading = skip_init_loading

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        if not self.never_unfreeze:
            for param in self.parameters():
                param.requires_grad = True

    def save_debug_data(self, save_dir, savename_suffix=''):
        """Template function for saving debug data"""
        pass

    def train_generator_update(self, epoch):
        pass
