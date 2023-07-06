from tinytorch import tensor
import torch
import numpy as np
import tinytorch.optim as optim
import tinytorch.nn.functional as F

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

#######################
# testing loss functions
#######################

# testing softmax function
x = tensor.Tensor([[1, 1], [2, 2], [3, 3]])
torch_t1 = torch.tensor([[1, 1], [2, 2], [3, 3]], dtype=torch.float64)
assert np.allclose(F.softmax(x, dim=1).data, torch.nn.functional.softmax(torch_t1, dim=1).data.numpy())

# testing log_softmax function
x = tensor.Tensor([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
torch_t1 = torch.tensor([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]], dtype=torch.float64)
assert np.allclose(F.log_softmax(x, dim=1).data, torch.nn.functional.log_softmax(torch_t1, dim=1).data.numpy())

# testing L1 loss function
x = tensor.Tensor([[1, 3, 6], [2, 4, 5]])
y = tensor.Tensor([[2, 4, 6], [3, 5, 7]])
torch_t1 = torch.tensor([[1, 3, 6], [2, 4, 5]], dtype=torch.float64)
torch_t2 = torch.tensor([[2, 4, 6], [3, 5, 7]], dtype=torch.float64)
assert np.allclose(F.l1_loss(x, y).data, torch.nn.functional.l1_loss(torch_t1, torch_t2).data.numpy())

# testing MSE loss function
x = tensor.Tensor([[1, 2], [3, 4], [5, 6]])
y = tensor.Tensor([[2, 3], [4, 5], [6, 7]])
torch_t1 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float64)
torch_t2 = torch.tensor([[2, 3], [4, 5], [6, 7]], dtype=torch.float64)
assert np.allclose(F.mse_loss(x, y).data, torch.nn.functional.mse_loss(torch_t1, torch_t2).data.numpy())

# testing NLL loss function
x = tensor.Tensor([[1, 3, 6], [2, 4, 5]])
y = tensor.Tensor([0, 1])
log_probs = F.log_softmax(x, dim=1)
torch_t1 = torch.tensor([[1, 3, 6], [2, 4, 5]], dtype=torch.float64)
torch_t2 = torch.tensor([0, 1], dtype=torch.int64)
torch_log_probs = torch.nn.functional.log_softmax(torch_t1, dim=1)
assert np.allclose(F.nll_loss(log_probs, y).data, torch.nn.functional.nll_loss(torch_log_probs, torch_t2).data.numpy())

# testing Cross Entropy loss function
x = tensor.Tensor([[1, 3, 6], [2, 4, 5]])
y = tensor.Tensor([0, 1])
torch_t1 = torch.tensor([[1, 3, 6], [2, 4, 5]], dtype=torch.float64)
torch_t2 = torch.tensor([0, 1], dtype=torch.int64)
assert np.allclose(F.cross_entropy(x, y).data, torch.nn.functional.cross_entropy(torch_t1, torch_t2).data.numpy())

# testing softmax function on 3D data
x = tensor.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
torch_t1 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float64)
assert np.allclose(F.softmax(x, dim=1).data, torch.nn.functional.softmax(torch_t1, dim=1).data.numpy())

# testing log_softmax function on 3D data
x = tensor.Tensor([[[0.2, 0.3], [0.4, 0.5]], [[0.6, 0.7], [0.8, 0.9]]])
torch_t1 = torch.tensor([[[0.2, 0.3], [0.4, 0.5]], [[0.6, 0.7], [0.8, 0.9]]], dtype=torch.float64)
assert np.allclose(F.log_softmax(x, dim=1).data, torch.nn.functional.log_softmax(torch_t1, dim=1).data.numpy())

# testing L1 loss function on 3D data
x = tensor.Tensor([[[1, 3], [2, 4]], [[5, 7], [6, 8]]])
y = tensor.Tensor([[[2, 4], [3, 5]], [[6, 8], [7, 9]]])
torch_t1 = torch.tensor([[[1, 3], [2, 4]], [[5, 7], [6, 8]]], dtype=torch.float64)
torch_t2 = torch.tensor([[[2, 4], [3, 5]], [[6, 8], [7, 9]]], dtype=torch.float64)
assert np.allclose(F.l1_loss(x, y).data, torch.nn.functional.l1_loss(torch_t1, torch_t2).data.numpy())

# testing MSE loss function on 3D data
x = tensor.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
y = tensor.Tensor([[[2, 3], [4, 5]], [[6, 7], [8, 9]]])
torch_t1 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float64)
torch_t2 = torch.tensor([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=torch.float64)
assert np.allclose(F.mse_loss(x, y).data, torch.nn.functional.mse_loss(torch_t1, torch_t2).data.numpy())


# backprop over NLL loss
torch_t1 = torch.tensor([[1, 3, 6, 4], [2, 4, 5, 4], [2, 3, 1, 5]], dtype=torch.float64, requires_grad=True)
torch_t2 = torch.tensor([0, 1, 2], dtype=torch.int64)
torch_log_probs = torch.nn.functional.log_softmax(torch_t1, dim=1)
torch_loss = torch.nn.functional.nll_loss(torch_log_probs, torch_t2, reduction='mean')
torch_loss.backward()

x = tensor.Tensor([[1, 3, 6, 4], [2, 4, 5, 4], [2, 3, 1, 5]], requires_grad=True)
y = tensor.Tensor([0, 1, 2])
loss_probs = F.log_softmax(x, dim=1)
loss = F.nll_loss(loss_probs, y, reduction='mean')
loss.backward()
assert np.allclose(x.grad, torch_t1.grad.data.numpy())

torch_t1 = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float64, requires_grad=True)
torch_t2 = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
torch_log_probs = torch.nn.functional.log_softmax(torch_t1, dim=2)
torch_loss = torch.nn.functional.nll_loss(torch_log_probs.flatten(0, 1), torch_t2.flatten(), reduction='mean')
torch_loss.backward()

x = tensor.Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], requires_grad=True)
y = tensor.Tensor([[0, 1], [0, 1]])
loss_probs = F.log_softmax(x, dim=2)
loss = F.nll_loss(loss_probs.reshape((-1, loss_probs.shape[-1])), y.flatten(), reduction='mean')
loss.backward()
assert np.allclose(x.grad, torch_t1.grad.data.numpy())
