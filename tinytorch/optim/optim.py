# base class for all optimizers
from ..tensor import Tensor
from typing import List, Tuple
import numpy as np


class Optimizer:
    def __init__(self, parameters: List[Tensor], lr: float = 0.01):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for param in self.parameters:
            if not getattr(param, "requires_grad", True):
                continue # skip frozen params
            if param.grad is not None:
                param.grad = np.zeros_like(param.grad)
            else:
                raise ValueError(f"Parameter {param} does not have gradient")

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
    ):
        
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.maximize = maximize
        
        if not 0 <= self.momentum < 1:
            raise ValueError("Invalid momentum value. Must be in range [0, 1]")
        if not 0 <= self.dampening <= 1:
            raise ValueError("Invalid dampening value. Must be in range [0, 1]")
        
        self.velocity = [
            np.zeros_like(p.data) if getattr(p, "requires_grad", True) else None for p in parameters
        ]
    
    def step(self):
        for i, param in enumerate(self.parameters):
            if not getattr(param, "requires_grad", True):
                continue
            
            if param.grad is None:
                raise ValueError(f"Parameter {param} does not have a gradient")
            
            grad = param.grad
            
            if self.maximize:
                grad = -grad
                
            # regularization
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # momentum update
            if self.momentum != 0:
                self.velocity[i] = (
                    self.momentum * self.velocity[i] + (1-self.dampening)*grad
                )
                
                if self.nesterov:
                    grad = grad + self.momentum * self.velocity[i]
                else:
                    grad = self.velocity[i]
                    
            param.data -= self.lr * grad
            
class RMSprop(Optimizer):
    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
        maximize: bool = False,
    ):
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        self.maximize = maximize

        # buffers for each parameter (None for frozen)
        self.square_avg = [
            np.zeros_like(p.data) if getattr(p, "requires_grad", True) else None
            for p in parameters
        ]
        self.momentum_buffer = [
            np.zeros_like(p.data) if (momentum > 0 and getattr(p, "requires_grad", True)) else None
            for p in parameters
        ]
        self.grad_avg = [
            np.zeros_like(p.data) if (centered and getattr(p, "requires_grad", True)) else None
            for p in parameters
        ]

    def step(self):
        for i, param in enumerate(self.parameters):
            if not getattr(param, "requires_grad", True):
                continue  # skip frozen params

            if param.grad is None:
                raise ValueError(f"Parameter {param} does not have gradient")

            grad = param.grad

            # handle maximize (gradient ascent)
            if self.maximize:
                grad = -grad

            # apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            # update running average of squared gradients
            self.square_avg[i] = (
                self.alpha * self.square_avg[i] + (1 - self.alpha) * (grad ** 2)
            )

            # denominator (variance estimate)
            avg = self.square_avg[i]

            if self.centered:
                # update grad moving average
                self.grad_avg[i] = (
                    self.alpha * self.grad_avg[i] + (1 - self.alpha) * grad
                )
                avg = avg - self.grad_avg[i] ** 2

            denom = (avg + self.eps)**(0.5) # sqrt

            if self.momentum > 0:
                self.momentum_buffer[i] = (
                    self.momentum * self.momentum_buffer[i] + grad / denom
                )
                update = self.momentum_buffer[i]
            else:
                update = grad / denom

            # parameter update
            param.data -= self.lr * update
            
            
class Adam(Optimizer):
    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        maximize: bool = False,
    ):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.maximize = maximize

        # timestep counter
        self.t = 0

        # per-parameter state (None for frozen params)
        self.m = [
            np.zeros_like(p.data) if getattr(p, "requires_grad", True) else None
            for p in parameters
        ]
        self.v = [
            np.zeros_like(p.data) if getattr(p, "requires_grad", True) else None
            for p in parameters
        ]
        self.v_max = [
            np.zeros_like(p.data) if (amsgrad and getattr(p, "requires_grad", True)) else None
            for p in parameters
        ]

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            if not getattr(param, "requires_grad", True):
                continue  # skip frozen params

            if param.grad is None:
                raise ValueError(f"Parameter {param} does not have gradient")

            grad = param.grad

            # handle maximize (gradient ascent)
            if self.maximize:
                grad = -grad

            # weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            # update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            if self.amsgrad:
                # maintain max of v
                self.v_max[i] = np.maximum(self.v_max[i], self.v[i])
                v_hat = self.v_max[i] / (1 - self.beta2 ** self.t)

            # parameter update
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
