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
        # return F.mse_loss(input, target, reduction=self.reduction)
        # checking shape: if shape is (n, 1) or (1, n), then squeeze it
        if input.shape == [target.shape[0], 1]:
            input = input.squeeze()
        return F.mse_loss(input, target, reduction=self.reduction)

    def __repr__(self):
        return f"MSELoss(reduction={self.reduction})"
    
class NLLLoss:
    """
    Negative Log Likelihood Loss
    """
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, input: tensor.Tensor, target: tensor.Tensor):
        return F.nll_loss(input, target, reduction=self.reduction)

    def __repr__(self):
        return f"NLLLoss(reduction={self.reduction})"
    
class CrossEntropyLoss:
    """
    Cross Entropy Loss
    """
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, input: tensor.Tensor, target: tensor.Tensor):
        return F.cross_entropy(input, target, reduction=self.reduction)

    def __repr__(self):
        return f"CrossEntropyLoss(reduction={self.reduction})"
    
class KLDivLoss:
    """
    KL Divergence
    """
    def __init__(self, reduction="mean", log_target=False):
        self.reduction = reduction
        self.log_target = log_target
    
    def __call__(self, input: tensor.Tensor, target: tensor.Tensor):
        return F.kl_div(input, target, reduction=self.reduction, log_target=self.log_target)
    
    def __repr__(self):
        return f"KLDivLoss(reduction={self.reduction}, log_target={self.log_target})"
    