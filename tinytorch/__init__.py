"""
Top-level namespace for TinyTorch.
"""

from .tensor import Tensor
from .util import save, load

# ---- nn sub-package -------------------------------------------------
from . import nn
from .nn import functional as F
from .nn.modules.module import Module
from .nn.modules.linear import Linear
from .nn.modules.activation import ReLU, Tanh, Sigmoid, Softmax, LogSoftmax
from .nn.modules.loss import (
    L1Loss,
    MSELoss,
    NLLLoss,
    CrossEntropyLoss,
    KLDivLoss,
)

# ---- optim sub-package ----------------------------------------------
from . import optim
from .optim.optim import (
    Optimizer,
    SGD,
    RMSprop,
    Adam,
)

# What will be imported on `from tinytorch import *`
__all__ = [
    "Tensor",
    "save",
    "load",
    "nn",
    "F",
    "Module",
    "Linear",
    "ReLU",
    "Tanh",
    "Sigmoid",
    "Softmax",
    "LogSoftmax",
    "L1Loss",
    "CrossEntropyLoss",
    "MSELoss",
    "NLLLoss",
    "KLDivLoss",
    # optim
    "optim",
    "Optimizer",
    "SGD",
    "RMSprop",
    "Adam",
]
