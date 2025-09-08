import numpy as np
import torch
import torch.nn.functional as F
import traceback
import sys

from tinytorch.tensor import Tensor
from tinytorch.nn import L1Loss, MSELoss, NLLLoss, CrossEntropyLoss, KLDivLoss

DTYPE = np.float64
TORCH_DTYPE = torch.float64

def np_allclose(a, b, rtol=1e-5, atol=1e-6):
    return np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), rtol=rtol, atol=atol)

def torch_from(np_array, requires_grad=False):
    return torch.tensor(np_array, dtype=TORCH_DTYPE, requires_grad=requires_grad)

def run_test(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        traceback.print_exc()
        return False


# ------------------------------
# Simple tests (your originals)
# ------------------------------

def test_l1_loss_mean():
    input_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=DTYPE)
    target_np = np.array([1.5, 1.8, 3.2, 3.9], dtype=DTYPE)

    input_tensor = Tensor(input_np, requires_grad=True)
    target_tensor = Tensor(target_np, requires_grad=False)
    loss_fn = L1Loss(reduction="mean")
    loss = loss_fn(input_tensor, target_tensor)
    loss.backward()

    input_torch = torch_from(input_np, requires_grad=True)
    target_torch = torch_from(target_np, requires_grad=False)
    loss_torch = F.l1_loss(input_torch, target_torch, reduction='mean')
    loss_torch.backward()

    assert np_allclose(loss.data, loss_torch.detach().numpy())
    assert np_allclose(input_tensor.grad, input_torch.grad.numpy())

def test_l1_loss_sum():
    input_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=DTYPE)
    target_np = np.array([[1.2, 1.9], [2.8, 4.1]], dtype=DTYPE)

    input_tensor = Tensor(input_np, requires_grad=True)
    target_tensor = Tensor(target_np, requires_grad=False)
    loss_fn = L1Loss(reduction="sum")
    loss = loss_fn(input_tensor, target_tensor)
    loss.backward()

    input_torch = torch_from(input_np, requires_grad=True)
    target_torch = torch_from(target_np, requires_grad=False)
    loss_torch = F.l1_loss(input_torch, target_torch, reduction='sum')
    loss_torch.backward()

    assert np_allclose(loss.data, loss_torch.detach().numpy())
    assert np_allclose(input_tensor.grad, input_torch.grad.numpy())

def test_mse_loss_mean():
    input_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=DTYPE)
    target_np = np.array([1.5, 1.8, 3.2, 3.9], dtype=DTYPE)

    input_tensor = Tensor(input_np, requires_grad=True)
    target_tensor = Tensor(target_np, requires_grad=False)
    loss_fn = MSELoss(reduction="mean")
    loss = loss_fn(input_tensor, target_tensor)
    loss.backward()

    input_torch = torch_from(input_np, requires_grad=True)
    target_torch = torch_from(target_np, requires_grad=False)
    loss_torch = F.mse_loss(input_torch, target_torch, reduction='mean')
    loss_torch.backward()

    assert np_allclose(loss.data, loss_torch.detach().numpy())
    assert np_allclose(input_tensor.grad, input_torch.grad.numpy())

def test_mse_loss_squeeze():
    input_np = np.array([[1.0], [2.0], [3.0]], dtype=DTYPE)
    target_np = np.array([1.5, 1.8, 3.2], dtype=DTYPE)

    input_tensor = Tensor(input_np, requires_grad=True)
    target_tensor = Tensor(target_np, requires_grad=False)
    loss_fn = MSELoss(reduction="mean")
    loss = loss_fn(input_tensor, target_tensor)
    loss.backward()

    expected_loss = np.mean((input_np.squeeze() - target_np) ** 2)
    assert np_allclose(loss.data, expected_loss)

def test_cross_entropy_loss():
    input_np = np.array([[2.0, 1.0, 0.5], [0.5, 2.5, 1.0]], dtype=DTYPE)
    target_np = np.array([0, 1], dtype=np.int64)

    input_tensor = Tensor(input_np, requires_grad=True)
    target_tensor = Tensor(target_np.astype(DTYPE), requires_grad=False)
    loss_fn = CrossEntropyLoss(reduction="mean")
    loss = loss_fn(input_tensor, target_tensor)
    loss.backward()

    input_torch = torch_from(input_np, requires_grad=True)
    target_torch = torch.tensor(target_np, dtype=torch.long)
    loss_torch = F.cross_entropy(input_torch, target_torch, reduction='mean')
    loss_torch.backward()

    assert np_allclose(loss.data, loss_torch.detach().numpy(), rtol=1e-4, atol=1e-5)
    assert np_allclose(input_tensor.grad, input_torch.grad.numpy(), rtol=1e-4, atol=1e-5)

def test_nll_loss():
    input_np = np.log(np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=DTYPE))
    target_np = np.array([0, 1], dtype=np.int64)

    input_tensor = Tensor(input_np, requires_grad=True)
    target_tensor = Tensor(target_np.astype(DTYPE), requires_grad=False)
    loss_fn = NLLLoss(reduction="mean")
    loss = loss_fn(input_tensor, target_tensor)
    loss.backward()

    input_torch = torch_from(input_np, requires_grad=True)
    target_torch = torch.tensor(target_np, dtype=torch.long)
    loss_torch = F.nll_loss(input_torch, target_torch, reduction='mean')
    loss_torch.backward()

    assert np_allclose(loss.data, loss_torch.detach().numpy(), rtol=1e-4, atol=1e-5)

def test_kl_div_loss():
    input_np = np.log(np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]], dtype=DTYPE))
    target_np = np.array([[0.5, 0.4, 0.1], [0.3, 0.6, 0.1]], dtype=DTYPE)

    # mini torch impl
    input_tensor = Tensor(input_np, requires_grad=True)
    target_tensor = Tensor(target_np, requires_grad=False)
    loss_fn = KLDivLoss(reduction="mean", log_target=False)
    loss = loss_fn(input_tensor, target_tensor)
    loss.backward()

    # PyTorch ground truth
    input_torch = torch_from(input_np, requires_grad=True)
    target_torch = torch_from(target_np, requires_grad=False)
    loss_torch = F.kl_div(input_torch, target_torch, reduction="mean", log_target=False)
    loss_torch.backward()

    # compare losses + grads
    assert np_allclose(loss.data, loss_torch.detach().numpy(), rtol=1e-4, atol=1e-5)
    assert np_allclose(input_tensor.grad, input_torch.grad.numpy(), rtol=1e-4, atol=1e-5)


def test_loss_none_reduction():
    input_np = np.array([1.0, 2.0, 3.0], dtype=DTYPE)
    target_np = np.array([1.2, 1.9, 2.8], dtype=DTYPE)

    input_tensor = Tensor(input_np, requires_grad=True)
    target_tensor = Tensor(target_np, requires_grad=False)

    loss_fn = L1Loss(reduction="none")
    loss = loss_fn(input_tensor, target_tensor)

    expected = np.abs(input_np - target_np)
    assert np_allclose(loss.data, expected)
    assert loss.data.shape == input_np.shape

def test_loss_sum_reduction():
    input_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=DTYPE)
    target_np = np.array([[1.1, 1.9], [2.9, 4.1]], dtype=DTYPE)

    input_tensor = Tensor(input_np, requires_grad=True)
    target_tensor = Tensor(target_np, requires_grad=False)

    loss_fn = MSELoss(reduction="sum")
    loss = loss_fn(input_tensor, target_tensor)

    expected = np.sum((input_np - target_np) ** 2)
    assert np_allclose(loss.data, expected)


# ------------------------------
# Complex tests (extended cases)
# ------------------------------

def test_l1_loss_mean_complex():
    np.random.seed(42)
    input_np = np.random.randn(3, 4, 5) * 2.0
    target_np = np.random.randn(3, 4, 5) * 2.0

    input_tensor = Tensor(input_np, requires_grad=True)
    target_tensor = Tensor(target_np, requires_grad=False)
    loss_fn = L1Loss(reduction="mean")
    loss = loss_fn(input_tensor, target_tensor)
    loss.backward()

    input_torch = torch_from(input_np, requires_grad=True)
    target_torch = torch_from(target_np, requires_grad=False)
    loss_torch = F.l1_loss(input_torch, target_torch, reduction="mean")
    loss_torch.backward()

    assert np_allclose(loss.data, loss_torch.detach().numpy())
    assert np_allclose(input_tensor.grad, input_torch.grad.numpy())

def test_mse_loss_sum_complex():
    np.random.seed(0)
    input_np = np.random.uniform(-3, 3, (2, 3, 4))
    target_np = np.random.uniform(-3, 3, (2, 3, 4))

    input_tensor = Tensor(input_np, requires_grad=True)
    target_tensor = Tensor(target_np, requires_grad=False)
    loss_fn = MSELoss(reduction="sum")
    loss = loss_fn(input_tensor, target_tensor)
    loss.backward()

    input_torch = torch_from(input_np, requires_grad=True)
    target_torch = torch_from(target_np, requires_grad=False)
    loss_torch = F.mse_loss(input_torch, target_torch, reduction="sum")
    loss_torch.backward()

    assert np_allclose(loss.data, loss_torch.detach().numpy())
    assert np_allclose(input_tensor.grad, input_torch.grad.numpy())

def test_cross_entropy_loss_complex():
    np.random.seed(123)
    input_np = np.random.randn(5, 10) * 5.0
    target_np = np.random.randint(0, 10, size=(5,))

    input_tensor = Tensor(input_np, requires_grad=True)
    target_tensor = Tensor(target_np.astype(DTYPE), requires_grad=False)
    loss_fn = CrossEntropyLoss(reduction="mean")
    loss = loss_fn(input_tensor, target_tensor)
    loss.backward()

    input_torch = torch_from(input_np, requires_grad=True)
    target_torch = torch.tensor(target_np, dtype=torch.long)
    loss_torch = F.cross_entropy(input_torch, target_torch, reduction="mean")
    loss_torch.backward()

    assert np_allclose(loss.data, loss_torch.detach().numpy(), rtol=1e-4, atol=1e-5)
    assert np_allclose(input_tensor.grad, input_torch.grad.numpy(), rtol=1e-4, atol=1e-5)


def test_kl_div_loss_complex():
    np.random.seed(7)
    target_np = np.random.dirichlet(np.ones(6), size=4)
    input_np = np.log(target_np + 1e-8)

    # mini torch impl
    input_tensor = Tensor(input_np, requires_grad=True)
    target_tensor = Tensor(target_np, requires_grad=False)
    loss_fn = KLDivLoss(reduction="batchmean", log_target=False)
    loss = loss_fn(input_tensor, target_tensor)
    loss.backward()

    # PyTorch ground truth
    input_torch = torch_from(input_np, requires_grad=True)
    target_torch = torch_from(target_np, requires_grad=False)
    loss_torch = F.kl_div(input_torch, target_torch, reduction="batchmean", log_target=False)
    loss_torch.backward()

    # compare losses + grads
    assert np_allclose(loss.data, loss_torch.detach().numpy(), rtol=1e-4, atol=1e-5)
    assert np_allclose(input_tensor.grad, input_torch.grad.numpy(), rtol=1e-4, atol=1e-5)


# ------------------------------
# Main
# ------------------------------

def main():
    torch.set_default_dtype(TORCH_DTYPE)
    tests = [
        ("l1_loss_mean", test_l1_loss_mean),
        ("l1_loss_sum", test_l1_loss_sum),
        ("mse_loss_mean", test_mse_loss_mean),
        ("mse_loss_squeeze", test_mse_loss_squeeze),
        ("cross_entropy_loss", test_cross_entropy_loss),
        ("nll_loss", test_nll_loss),
        ("kl_div_loss", test_kl_div_loss),
        ("loss_none_reduction", test_loss_none_reduction),
        ("loss_sum_reduction", test_loss_sum_reduction),

        # Complex cases
        ("l1_loss_mean_complex", test_l1_loss_mean_complex),
        ("mse_loss_sum_complex", test_mse_loss_sum_complex),
        ("cross_entropy_loss_complex", test_cross_entropy_loss_complex),
        ("kl_div_loss_complex", test_kl_div_loss_complex),
    ]

    passed = 0
    for name, fn in tests:
        if run_test(name, fn):
            passed += 1

    total = len(tests)
    print(f"\nSummary: {passed}/{total} tests passed.")
    if passed != total:
        sys.exit(1)

if __name__ == "__main__":
    main()
