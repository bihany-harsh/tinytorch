from .module import Module
from tinytorch import tensor

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = tensor.randn((out_features, in_features), requires_grad=True)
        self.bias = tensor.randn((out_features,), requires_grad=True) if bias else None
        
    def forward(self, x):
        y = x @ self.weight.T
        return y + self.bias if self.bias is not None else y
    
    def __repr__(self) -> str:
        return (f"Linear(in_features={self.weight.shape[1]}, "
                f"out_features={self.weight.shape[0]}, "
                f"bias={self.bias is not None})")