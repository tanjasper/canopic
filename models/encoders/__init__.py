"""
How to design an encoder:
1. init function must always take opt as the first argument (for consistency)
2. Following opt, init function must list all user parameters as optional parameters. These will be passed in by the
    user as a dictionary
    If you wish to require certain parameters to be passed in, a workaround is to set its default as None (or other
    value that the user will never input) and raise an error in the init function if such value is None (i.e., no value
    input)
3. Encoder class must inherit from BaseEncoder

"""

from models.encoders.convolutional import Convolution
from models.encoders.quantization import LearnQuantize, FixedQuantize
from models.encoders.combination import ConvMxPQuantize
from models.encoders.scratch import ScratchConv, ScratchThreeConv
from models.encoders.basic import PassThrough, Normalization, GaussianBlur
from models.encoders.pooling import MaxPool, Upsample, AveragePool
from models.encoders.resize import FasterRCNNResize
