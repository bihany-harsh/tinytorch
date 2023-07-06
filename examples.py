from tinytorch import tensor
import torch
import numpy as np
import tinytorch.optim as optim
import tinytorch.nn.functional as F
import tinytorch.nn.modules.loss as loss_module
import tinytorch.nn.modules.activation as activation_module

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

# testing softmax function
x = tensor.Tensor([[1, 1], [2, 2], [3, 3]])
torch_t1 = torch.tensor([[1, 1], [2, 2], [3, 3]], dtype=torch.float64)
assert np.allclose(
    F.softmax(x, dim=1).data, torch.nn.functional.softmax(torch_t1, dim=1).data.numpy()
)

# testing log_softmax function
x = tensor.Tensor([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
torch_t1 = torch.tensor([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]], dtype=torch.float64)
assert np.allclose(
    F.log_softmax(x, dim=1).data,
    torch.nn.functional.log_softmax(torch_t1, dim=1).data.numpy(),
)

# testing L1 loss function
x = tensor.Tensor([[1, 3, 6], [2, 4, 5]])
y = tensor.Tensor([[2, 4, 6], [3, 5, 7]])
torch_t1 = torch.tensor([[1, 3, 6], [2, 4, 5]], dtype=torch.float64)
torch_t2 = torch.tensor([[2, 4, 6], [3, 5, 7]], dtype=torch.float64)
assert np.allclose(
    F.l1_loss(x, y).data, torch.nn.functional.l1_loss(torch_t1, torch_t2).data.numpy()
)

# testing MSE loss function
x = tensor.Tensor([[1, 2], [3, 4], [5, 6]])
y = tensor.Tensor([[2, 3], [4, 5], [6, 7]])
torch_t1 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float64)
torch_t2 = torch.tensor([[2, 3], [4, 5], [6, 7]], dtype=torch.float64)
assert np.allclose(
    F.mse_loss(x, y).data, torch.nn.functional.mse_loss(torch_t1, torch_t2).data.numpy()
)

# testing NLL loss function
x = tensor.Tensor([[1, 3, 6], [2, 4, 5]])
y = tensor.Tensor([0, 1])
log_probs = F.log_softmax(x, dim=1)
torch_t1 = torch.tensor([[1, 3, 6], [2, 4, 5]], dtype=torch.float64)
torch_t2 = torch.tensor([0, 1], dtype=torch.int64)
torch_log_probs = torch.nn.functional.log_softmax(torch_t1, dim=1)
assert np.allclose(
    F.nll_loss(log_probs, y).data,
    torch.nn.functional.nll_loss(torch_log_probs, torch_t2).data.numpy(),
)

# testing Cross Entropy loss function
x = tensor.Tensor([[1, 3, 6], [2, 4, 5]])
y = tensor.Tensor([0, 1])
torch_t1 = torch.tensor([[1, 3, 6], [2, 4, 5]], dtype=torch.float64)
torch_t2 = torch.tensor([0, 1], dtype=torch.int64)
assert np.allclose(
    F.cross_entropy(x, y).data,
    torch.nn.functional.cross_entropy(torch_t1, torch_t2).data.numpy(),
)

# testing softmax function on 3D data
x = tensor.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
torch_t1 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float64)
assert np.allclose(
    F.softmax(x, dim=1).data, torch.nn.functional.softmax(torch_t1, dim=1).data.numpy()
)

# testing log_softmax function on 3D data
x = tensor.Tensor([[[0.2, 0.3], [0.4, 0.5]], [[0.6, 0.7], [0.8, 0.9]]])
torch_t1 = torch.tensor(
    [[[0.2, 0.3], [0.4, 0.5]], [[0.6, 0.7], [0.8, 0.9]]], dtype=torch.float64
)
assert np.allclose(
    F.log_softmax(x, dim=1).data,
    torch.nn.functional.log_softmax(torch_t1, dim=1).data.numpy(),
)

# testing L1 loss function on 3D data
x = tensor.Tensor([[[1, 3], [2, 4]], [[5, 7], [6, 8]]])
y = tensor.Tensor([[[2, 4], [3, 5]], [[6, 8], [7, 9]]])
torch_t1 = torch.tensor([[[1, 3], [2, 4]], [[5, 7], [6, 8]]], dtype=torch.float64)
torch_t2 = torch.tensor([[[2, 4], [3, 5]], [[6, 8], [7, 9]]], dtype=torch.float64)
assert np.allclose(
    F.l1_loss(x, y).data, torch.nn.functional.l1_loss(torch_t1, torch_t2).data.numpy()
)

# testing MSE loss function on 3D data
x = tensor.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
y = tensor.Tensor([[[2, 3], [4, 5]], [[6, 7], [8, 9]]])
torch_t1 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float64)
torch_t2 = torch.tensor([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=torch.float64)
assert np.allclose(
    F.mse_loss(x, y).data, torch.nn.functional.mse_loss(torch_t1, torch_t2).data.numpy()
)

# backprop over MSE loss
torch_t1 = torch.tensor(
    [[1, 2], [3, 4], [5, 6]], dtype=torch.float64, requires_grad=True
)
torch_t2 = torch.tensor([[2, 3], [4, 5], [6, 7]], dtype=torch.float64)
torch_loss = torch.nn.functional.mse_loss(torch_t1, torch_t2, reduction="mean")
torch_loss.backward()

x = tensor.Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True, dtype=np.float64)
y = tensor.Tensor([[2, 3], [4, 5], [6, 7]])
loss = F.mse_loss(x, y)
loss.backward()
assert np.allclose(x.grad.data, torch_t1.grad.data.numpy())

# backprop over NLL loss
torch_t1 = torch.tensor(
    [[1, 3, 6, 4], [2, 4, 5, 4], [2, 3, 1, 5]], dtype=torch.float64, requires_grad=True
)
torch_t2 = torch.tensor([0, 1, 2], dtype=torch.int64)
torch_log_probs = torch.nn.functional.log_softmax(torch_t1, dim=1)
torch_loss = torch.nn.functional.nll_loss(torch_log_probs, torch_t2, reduction="mean")
torch_loss.backward()

x = tensor.Tensor([[1, 3, 6, 4], [2, 4, 5, 4], [2, 3, 1, 5]], requires_grad=True)
y = tensor.Tensor([0, 1, 2])
loss_probs = F.log_softmax(x, dim=1)
loss = F.nll_loss(loss_probs, y, reduction="mean")
loss.backward()
assert np.allclose(x.grad, torch_t1.grad.data.numpy())

torch_t1 = torch.tensor(
    [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
    dtype=torch.float64,
    requires_grad=True,
)
torch_t2 = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
torch_log_probs = torch.nn.functional.log_softmax(torch_t1, dim=2)
torch_loss = torch.nn.functional.nll_loss(
    torch_log_probs.flatten(0, 1), torch_t2.flatten(), reduction="mean"
)
torch_loss.backward()

x = tensor.Tensor(
    [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], requires_grad=True
)
y = tensor.Tensor([[0, 1], [0, 1]])
loss_probs = F.log_softmax(x, dim=2)
loss = F.nll_loss(
    loss_probs.reshape((-1, loss_probs.shape[-1])), y.flatten(), reduction="mean"
)
loss.backward()
assert np.allclose(x.grad, torch_t1.grad.data.numpy())

# backprop over Cross Entropy loss
torch_t1 = torch.tensor(
    [[1, 3, 6, 4], [2, 4, 5, 4], [2, 3, 1, 5]], dtype=torch.float64, requires_grad=True
)
target = torch.tensor([0, 1, 2], dtype=torch.int64)
torch_loss = torch.nn.functional.cross_entropy(torch_t1, target, reduction="mean")
torch_loss.backward()

x = tensor.Tensor([[1, 3, 6, 4], [2, 4, 5, 4], [2, 3, 1, 5]], requires_grad=True)
y = tensor.Tensor([0, 1, 2])
loss = F.cross_entropy(x, y, reduction="mean")
loss.backward()
assert np.allclose(x.grad, torch_t1.grad.data.numpy())

# backprop over losses as classes: NLLLoss
torch_t1 = torch.tensor(
    [[1, 3, 6, 4], [2, 4, 5, 4], [2, 3, 1, 5]], dtype=torch.float64, requires_grad=True
)
torch_t2 = torch.tensor([0, 1, 2], dtype=torch.int64)
torch_loss = torch.nn.NLLLoss(reduction="mean")(
    torch.nn.functional.log_softmax(torch_t1, dim=1), torch_t2
)
torch_loss.backward()

x = tensor.Tensor([[1, 3, 6, 4], [2, 4, 5, 4], [2, 3, 1, 5]], requires_grad=True)
y = tensor.Tensor([0, 1, 2])
loss = loss_module.NLLLoss(reduction="mean")(F.log_softmax(x, dim=1), y)
loss.backward()
assert np.allclose(x.grad, torch_t1.grad.data.numpy())

# backprop over losses as classes: CrossEntropyLoss
torch_t1 = torch.tensor(
    [[1, 3, 6, 4], [2, 4, 5, 4], [2, 3, 1, 5]], dtype=torch.float64, requires_grad=True
)
target = torch.tensor([0, 1, 2], dtype=torch.int64)
torch_loss = torch.nn.CrossEntropyLoss(reduction="mean")(torch_t1, target)
torch_loss.backward()

x = tensor.Tensor([[1, 3, 6, 4], [2, 4, 5, 4], [2, 3, 1, 5]], requires_grad=True)
y = tensor.Tensor([0, 1, 2])
loss = loss_module.CrossEntropyLoss(reduction="mean")(x, y)
loss.backward()
assert np.allclose(x.grad, torch_t1.grad.data.numpy())

#######################
# testing matmul and backprop through matmul
#######################
x = tensor.Tensor([[1, 2], [3, 4]], requires_grad=True)
W = tensor.Tensor([[1], [2]], requires_grad=True)
target = tensor.Tensor([[4], [8]])
loss = F.mse_loss(x @ W, target)
loss.backward()

torch_x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64, requires_grad=True)
torch_W = torch.tensor([[1], [2]], dtype=torch.float64, requires_grad=True)
torch_target = torch.tensor([[4], [8]], dtype=torch.float64)
torch_loss = torch.nn.functional.mse_loss(torch_x @ torch_W, torch_target)
torch_loss.backward()

assert np.allclose(x.grad, torch_x.grad.data.numpy())

# a tougher Test
x = tensor.Tensor(
    [
        [1, 2, 3, 4], 
        [2, 3, 4, 5], 
    ],
    requires_grad=True,
)
W = tensor.Tensor(
    [[1, 2, 1, 2, 1], [2, 3, 2, 3, 2], [3, 4, 3, 4, 3], [4, 5, 4, 5, 4]],
    requires_grad=True,
)
target = tensor.Tensor([0, 1], dtype=np.int32)
logits = x @ W
loss = F.cross_entropy(logits, target)
loss.backward()

torch_x = torch.tensor(x.data, dtype=torch.float64, requires_grad=True)
torch_W = torch.tensor(W.data, dtype=torch.float64, requires_grad=True)
torch_target = torch.tensor(target.data, dtype=torch.int64)
torch_logits = torch_x @ torch_W
torch_loss = torch.nn.functional.cross_entropy(torch_logits, torch_target)
torch_loss.backward()

assert np.allclose(x.grad, torch_x.grad.data.numpy())

# just testing the ReLU
# x = tensor.Tensor([[1, -1], [-1, 1]], requires_grad=True)
# act = activation_module.ReLU()
# y = act(x)
# print(y)