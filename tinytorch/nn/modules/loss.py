from tinytorch import tensor
import tinytorch.nn.functional as F

class L1Loss:
    """
    L1 Loss
    """
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, input: tensor.Tensor, target: tensor.Tensor):
        return F.l1_loss(input, target, reduction=self.reduction)

    def __repr__(self):
        return f"L1Loss(reduction={self.reduction})"

class MSELoss:
    """
    Mean Squared Error Loss
    """
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, input: tensor.Tensor, target: tensor.Tensor):
        return F.mse_loss(input, target, reduction=self.reduction)

    def __repr__(self):
        return f"MSELoss(reduction={self.reduction})"
    