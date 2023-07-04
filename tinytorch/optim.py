# base class for all optimizers
from .tensor import Tensor
from typing import List
import numpy as np


class Optimizer:
    def __init__(self, parameters: List[Tensor], lr: float = 0.01):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad = np.zeros_like(param.grad)
            else:
                raise ValueError(f"Parameter {param} does not have gradient")

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(
        self, parameters: List[Tensor], lr: float = 0.01, momentum: float = 0.01
    ):
        super().__init__(parameters, lr)
        self.momentum = momentum
        if not 0 <= self.momentum < 1:
            raise ValueError("Invalid momentum value. Must be in range [0, 1]")
        self.velocity = [0 for _ in range(len(parameters))]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                self.velocity[i] = (
                    self.momentum * self.velocity[i] + self.lr * param.grad
                )
                param.data -= self.velocity[i]
            else:
                raise ValueError(f"Parameter {param} does not have gradient")
