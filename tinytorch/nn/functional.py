import numpy as np
from tinytorch import tensor

def softmax(input: tensor.Tensor, dim=0):
    """
    Args:
        input shape: (d1, d2, d3 ...)
        dim: the dimension to apply softmax
    Returns:
        output shape: (d1, d2, d3 ...) where the values along dim sum to 1
    """
    if dim >= len(input.shape):
        raise ValueError(f"dim should be less than the number of dimensions of input {len(input.shape)}")
    exp = input.exp()
    exp /= exp.sum(dim=dim, keepdim=True)
    
    # def _grad_fn():
    #     input.grad += (exp * (1 - exp) * exp.grad).data

    def _grad_fn():
        # Move the softmax dimension to the end
        s = np.moveaxis(exp.data, dim, -1)
        grad_exp = np.moveaxis(exp.grad, dim, -1)
        jacobian = np.zeros(s.shape + s.shape[-1:])
        for indices in np.ndindex(s.shape[:-1]):
            s_slice = s[indices]
            jacobian[indices] = np.diagflat(s_slice) - np.outer(s_slice, s_slice)
        grad_input = np.einsum('...ij,...j->...i', jacobian, grad_exp)
        grad_input = np.moveaxis(grad_input, -1, dim)
        input.grad += grad_input

    exp._grad_fn = _grad_fn
    return exp

def log_softmax(input: tensor.Tensor, dim=0):
    softmax_output = softmax(input, dim=dim)
    return softmax_output.log()


def l1_loss(input: tensor.Tensor, target: tensor.Tensor, reduction="mean"):
    if input.shape != target.shape:
        raise ValueError("input and target must have the same shape")
    loss = (input - target).abs()
    if reduction == "mean":
        loss = loss.mean(dim=None, keepdim=True)
    elif reduction == "sum":
        loss = loss.sum(dim=None, keepdim=True)
    elif reduction == "none":
        pass
    else:
        raise ValueError("Invalid value for reduction")
    return loss


def mse_loss(input: tensor.Tensor, target: tensor.Tensor, reduction="mean"):
    if input.shape != target.shape:
        raise ValueError("input and target must have the same shape")
    loss = (input - target) ** 2
    if reduction == "mean":
        loss = loss.mean(dim=None, keepdim=False)
    elif reduction == "sum":
        loss = loss.sum(dim=None, keepdim=False)
    elif reduction == "none":
        pass
    else:
        raise ValueError("Invalid value for reduction")
    return loss


def nll_loss(input: tensor.Tensor, target: tensor.Tensor, reduction="mean"):
    """
    Args:
        input shape: (N, C, d1, d2...) : C is the number of classes
        target shape: (N, d1, d2...)
    Returns:
        loss shape: shape based on reduction
    """
    if input.shape[0] != target.shape[0]:
        raise ValueError("input and target must have the same batch size")
    if len(input.shape) != len(target.shape) + 1:
        raise ValueError(
            f"invalid input: {input.shape} and target shape: {target.shape}"
        )
    
    # arrays used as indices must be integer (or boolean)

    loss = input[tensor.arange(input.shape[0]), target.data.astype(int)]

    # loss.data should be a dimensionless array 

    if reduction == "mean":
        loss = -loss.mean(dim=None, keepdim=False)
    elif reduction == "sum":
        loss = -loss.sum(dim=None, keepdim=False)
    elif reduction == "none":
        pass
    else:
        raise ValueError("Invalid value for reduction")
    
    return loss

def cross_entropy(input: tensor.Tensor, target: tensor.Tensor, reduction="mean"):
    """
    Args:
        input shape: (N, C, d1, d2...) : C is the number of classes
        target shape: (N, d1, d2...)
    Returns:
        loss shape: shape based on reduction
    """
    log_softmax_output = log_softmax(input, dim=1)
    loss = nll_loss(log_softmax_output, target, reduction=reduction)
    return loss