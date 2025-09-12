import torch 
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(126, 256)
        self.params = nn.Parameter(torch.randn(128, 256))
    
    def forward(self, x):
        return x

mod = MyModule()
for n, p in mod.named_parameters():
    print(n)
    