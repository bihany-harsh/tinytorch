from tinytorch import tensor
import torch
import numpy as np
import tinytorch.optim as optim

t1 = tensor.Tensor([[1, 2, 3], [4, 5, 6]])
t2 = tensor.Tensor([1, 4, 7])
# print(t1 + t2)
# print(-t2)
# print(t1 / t2)
# print(t2 ** 2)

torch_t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
torch_t2 = torch.tensor([1, 4, 7])
# print(torch_t1 * torch_t2)
# print(-torch_t2)
# print(torch_t1 / torch_t2)
# print(torch_t2 ** 2)

arr = tensor.arange(10)
# print(arr)

#######################
# testing autograd
#######################

# x1 = tensor.Tensor([2.0, 4.0], requires_grad=True)
# w1 = tensor.Tensor([-1.0, 3.0], requires_grad=True)
# x2 = tensor.Tensor([0.0, 1.2], requires_grad=True)
# w2 = tensor.Tensor([-1.0, -1.5], requires_grad=True)
# b = tensor.Tensor([5.0, 7.0], requires_grad=True)
# y = x1*w1 + x2*w2 + b
# print(y)
# parameters = [w1, w2, b]
# sgd = optim.SGD(parameters=parameters, lr=0.01, momentum=0.0)
# sgd.zero_grad()
# y.backward()
# sgd.step()

# print(w1, w2, b)

#######################
# testing functionalities
#######################

# x = tensor.Tensor([[1, 2, 3], [4, 5, 6]])
# y = tensor.Tensor([1, 2, 5])
# ex_x, ex_y = tensor.brodcast_tensors(x, y)
# print(ex_x, "\n", ex_y)


import tinytorch.nn.functional as F

x = tensor.Tensor([[1, 3, 6], [2, 4, 5]])

torch_t1 = torch.tensor([[1, 3, 6], [2, 4, 5]], dtype=torch.float64)

assert np.allclose(F.softmax(x, dim=1).data, torch.nn.functional.softmax(torch_t1, dim=1).data.numpy())
assert np.allclose(F.log_softmax(x, dim=1).data, torch.nn.functional.log_softmax(torch_t1, dim=1).data.numpy())


#######################
# testing loss functions
#######################

# L1 loss
x = tensor.Tensor([[1, 3, 6], [2, 4, 5]])
y = tensor.Tensor([[1, 2, 5], [2, 4, 5]])

torch_t1 = torch.tensor([[1, 3, 6], [2, 4, 5]], dtype=torch.float64)
torch_t2 = torch.tensor([[1, 2, 5], [2, 4, 5]], dtype=torch.float64)

assert np.allclose(F.l1_loss(x, y).data, torch.nn.functional.l1_loss(torch_t1, torch_t2).data.numpy())

# MSE loss

assert np.allclose(F.mse_loss(x, y).data, torch.nn.functional.mse_loss(torch_t1, torch_t2).data.numpy())

# NLL loss

x = tensor.Tensor([[1, 3, 6], [2, 4, 5]])
y = tensor.Tensor([1, 2])
log_probs = F.log_softmax(x, dim=1)
torch_t1 = torch.tensor([[1, 3, 6], [2, 4, 5]], dtype=torch.float64)
torch_t2 = torch.tensor([1, 2], dtype=torch.int64)
torch_log_probs = torch.nn.functional.log_softmax(torch_t1, dim=1)

assert np.allclose(F.nll_loss(log_probs, y).data, torch.nn.functional.nll_loss(torch_log_probs, torch_t2).data.numpy())
