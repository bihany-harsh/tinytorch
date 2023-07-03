from tinytorch import tensor
import torch
import numpy as np

t1 = tensor.Tensor([[1, 2, 3], [4, 5, 6]])
t2 = tensor.Tensor([1, 4, 7])
print(t1 * t2)
print(-t2)
print(t1 / t2)
print(t2 ** 2)

torch_t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
torch_t2 = torch.tensor([1, 4, 7])
print(torch_t1 * torch_t2)
print(-torch_t2)
print(torch_t1 / torch_t2)
print(torch_t2 ** 2)

arr = tensor.arange(10)
print(arr)