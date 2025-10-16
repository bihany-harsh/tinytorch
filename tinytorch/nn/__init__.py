"""
Neural-network sub-package (tinytorch.nn).
"""

from . import functional
from .functional import *

from .modules.module import Module, Parameter
from .modules.linear import Linear
from .modules.activation import ReLU, Tanh, Sigmoid, Softmax, LogSoftmax
from .modules.loss import L1Loss, CrossEntropyLoss, MSELoss, NLLLoss, KLDivLoss

# Public API
__all__ = [
    # functional symbols
    *functional.__all__,
    # modules
    "Module",
    "Parameter",
    "Linear",
    "ReLU",
    "Tanh",
    "Sigmoid",
    "Softmax",
    "LogSoftmax",
    "MSELoss",
    "L1Loss",
    "CrossEntropyLoss",
    "NLLLoss",
    "KLDivLoss",
]
