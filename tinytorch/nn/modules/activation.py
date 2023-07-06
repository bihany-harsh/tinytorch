class ReLU:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        print(self.mask)
        out[self.mask] = 0

        def _grad_fn():
            grad = out.copy()
            grad[self.mask] = 0
            return grad        
        return out
    
    def __repr__(self):
        return 'ReLU()'
    
    def __call__(self, x):
        return self.forward(x)