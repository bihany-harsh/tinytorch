import numpy as np
import tinytorch.tensor as tensor

class Linear:
    def __init__(self, in_features, out_features, bias=True, random_state=None):
        self.in_features = in_features
        self.out_features = out_features
        if random_state is not None:
            np.random.seed(random_state)       

        if out_features == 1:
            self.weights = tensor.randn((in_features,), requires_grad=True)     
        else:
            self.weights = tensor.randn((out_features, in_features), requires_grad=True)
        if bias:
            self.bias = tensor.randn((out_features, ), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = x @ self.weights.T + self.bias
        else:
            out = x @ self.weights.T
        return out
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        return [self.weights] + ([self.bias] if self.bias is not None else [])
    
    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()