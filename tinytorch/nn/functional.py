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
    return exp / exp.sum(dim=dim, keepdim=True)


def log_softmax(input: tensor.Tensor, dim=0):
    # use the softmax function above
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
        loss = loss.mean(dim=None, keepdim=True)
    elif reduction == "sum":
        loss = loss.sum(dim=None, keepdim=True)
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
    
    # arrays used as indices must be integer (or boolean)s
    loss = tensor.Tensor(
        input.data[np.arange(input.shape[0]), target.data.astype(int)].reshape(-1),
        (input, target),
        requires_grad=input.requires_grad,
    )

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