import numpy as np
from tinytorch import tensor

def softmax(input: tensor.Tensor, dim=0):
    """
    Args:
        input shape: (d1, d2, d3 ...)
        dim: the dimension to apply softmax
    """
    exp = (input - input.max(dim=dim, keepdim=True)).exp()
    sum = exp.sum(dim=dim, keepdim=True)
    exp /= sum
    return exp

def log_softmax(input: tensor.Tensor, dim=0):
    softmax_output = softmax(input, dim=dim)
    return softmax_output.log()


# LOSS FUNCTIONS

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
    """
    if input.shape[0] != target.shape[0]:
        raise ValueError("input and target must have the same batch size")
    if len(input.shape) != len(target.shape) + 1:
        raise ValueError(
            f"invalid input: {input.shape} and target shape: {target.shape}"
        )
    
    # arrays used as indices must be integer (or boolean)

    row_indices = np.arange(input.shape[0])
    col_indices = target.data.astype(int)
    loss = input[row_indices, col_indices]


    # loss.data should be a dimensionless array
    if reduction == "mean":
        loss = -loss.mean(dim=None, keepdim=False)
    elif reduction == "sum":
        loss = -loss.sum(dim=None, keepdim=False)
    elif reduction == "none":
        loss = -loss
    else:
        raise ValueError("Invalid value for reduction")
    
    return loss

def cross_entropy(input: tensor.Tensor, target: tensor.Tensor, reduction="mean"):
    """
    Args:
        input shape: (N, C, d1, d2...) : C is the number of classes
        target shape: (N, d1, d2...)
    """
    log_softmax_output = log_softmax(input, dim=1)
    loss = nll_loss(log_softmax_output, target, reduction=reduction)
    return loss

def kl_div(input: tensor.Tensor, target: tensor.Tensor, reduction: str = "mean", log_target: bool = False):
    """
    Kullback-Leibler divergence.
    """
    if log_target:
        loss_elements = target.exp() * (target - input)
    else:
        non_zero_mask = target != 0
        safe_target_log = tensor.where(non_zero_mask, target.log(), 0)
        loss_unmasked = target * (safe_target_log - input)
        loss_elements = tensor.where(non_zero_mask, loss_unmasked, 0)

    if reduction == "mean":  # default
        loss = loss_elements.mean()
    elif reduction == "batchmean":  # mathematically correct
        loss = loss_elements.sum() / input.size(0)
    elif reduction == "sum":
        loss = loss_elements.sum()
    else:  # reduction == "none"
        loss = loss_elements
        
    return loss
    
# ACTIVATION FUNCTIONS

def relu(input: tensor.Tensor):
    out = tensor.where(input > 0, input, 0)
    return out

def tanh(input: tensor.Tensor):
    out = (input.exp() - (-input).exp()) / (input.exp() + (-input).exp())
    return out

def sigmoid(input: tensor.Tensor):
    out = (1.0 + (-input).exp())**(-1)
    return out