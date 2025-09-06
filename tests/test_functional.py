import numpy as np
import torch
import traceback
import sys

from tinytorch.tensor import Tensor, where
from tinytorch.nn.functional import (softmax, log_softmax, l1_loss, mse_loss, nll_loss, cross_entropy, relu, tanh, sigmoid)

DTYPE = np.float64
TORCH_DTYPE = torch.float64

def np_allclose(a, b, rtol=1e-6, atol=1e-8):
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


def test_softmax_forward_and_grad():
    x_np = np.random.randn(3, 4).astype(DTYPE)
    x = Tensor(x_np, requires_grad=True)
    out = softmax(x, dim=1)
    (out.sum()).backward()

    xt = torch_from(x_np, requires_grad=True)
    outt = torch.nn.functional.softmax(xt, dim=1)
    (outt.sum()).backward()

    assert np_allclose(out.data, outt.detach().numpy())
    assert np_allclose(x.grad, xt.grad.numpy())

def test_log_softmax_forward_and_grad():
    x_np = np.random.randn(2, 5).astype(DTYPE)
    x = Tensor(x_np, requires_grad=True)
    out = log_softmax(x, dim=1)
    (out.sum()).backward()

    xt = torch_from(x_np, requires_grad=True)
    outt = torch.nn.functional.log_softmax(xt, dim=1)
    (outt.sum()).backward()

    assert np_allclose(out.data, outt.detach().numpy())
    assert np_allclose(x.grad, xt.grad.numpy())

def test_l1_loss_forward_and_grad():
    input_np = np.random.randn(3, 4).astype(DTYPE)
    target_np = np.random.randn(3, 4).astype(DTYPE)
    for reduction in ["mean", "sum", "none"]:
        input = Tensor(input_np, requires_grad=True)
        target = Tensor(target_np, requires_grad=False)
        out = l1_loss(input, target, reduction=reduction)
        (out.sum()).backward()

        it = torch_from(input_np, requires_grad=True)
        tt = torch_from(target_np)
        outt = torch.nn.functional.l1_loss(it, tt, reduction=reduction)
        (outt.sum()).backward()

        assert np_allclose(out.data, outt.detach().numpy())
        assert np_allclose(input.grad, it.grad.numpy())

def test_mse_loss_forward_and_grad():
    input_np = np.random.randn(3, 4).astype(DTYPE)
    target_np = np.random.randn(3, 4).astype(DTYPE)
    for reduction in ["mean", "sum", "none"]:
        input = Tensor(input_np, requires_grad=True)
        target = Tensor(target_np, requires_grad=False)
        out = mse_loss(input, target, reduction=reduction)
        (out.sum()).backward()

        it = torch_from(input_np, requires_grad=True)
        tt = torch_from(target_np)
        outt = torch.nn.functional.mse_loss(it, tt, reduction=reduction)
        (outt.sum()).backward()

        assert np_allclose(out.data, outt.detach().numpy())
        assert np_allclose(input.grad, it.grad.numpy())

def test_nll_loss_forward_and_grad():
    input_np = np.random.randn(4, 3).astype(DTYPE)
    target_np = np.array([0, 2, 1, 1], dtype=np.int64)
    for reduction in ["mean", "sum", "none"]:
        input = Tensor(input_np, requires_grad=True)
        target = Tensor(target_np, requires_grad=False)
        out = nll_loss(input, target, reduction=reduction)

        if reduction == "none":
            out.backward(Tensor(np.ones_like(out.data)))
        else:
            out.backward()

        it = torch_from(input_np, requires_grad=True)
        tt = torch.tensor(target_np, dtype=torch.long)
        outt = torch.nn.functional.nll_loss(it, tt, reduction=reduction)

        if reduction == "none":
            outt.backward(torch.ones_like(outt))
        else:
            outt.backward()
            
        assert np_allclose(out.data, outt.detach().numpy())
        assert np_allclose(input.grad, it.grad.numpy())

def test_cross_entropy_forward_and_grad():
    input_np = np.random.randn(5, 4).astype(DTYPE)
    target_np = np.random.randint(0, 4, size=(5,), dtype=np.int64)
    for reduction in ["mean", "sum", "none"]:
        input = Tensor(input_np, requires_grad=True)
        target = Tensor(target_np, requires_grad=False)
        out = cross_entropy(input, target, reduction=reduction)
        if reduction == "none":
            out.backward(Tensor(np.ones_like(out.data)))
        else:
            out.backward()

        it = torch_from(input_np, requires_grad=True)
        tt = torch.tensor(target_np, dtype=torch.long)
        outt = torch.nn.functional.cross_entropy(it, tt, reduction=reduction)
        if reduction == "none":
            outt.backward(torch.ones_like(outt))
        else:
            outt.backward()

        assert np_allclose(out.data, outt.detach().numpy())
        assert np_allclose(input.grad, it.grad.numpy())

def test_relu_forward_and_grad():
    x_np = np.random.randn(6).astype(DTYPE)
    x = Tensor(x_np, requires_grad=True)
    out = relu(x)
    out.sum().backward()

    xt = torch_from(x_np, requires_grad=True)
    outt = torch.nn.functional.relu(xt)
    outt.sum().backward()

    assert np_allclose(out.data, outt.detach().numpy())
    assert np_allclose(x.grad, xt.grad.numpy())

def test_tanh_forward_and_grad():
    x_np = np.random.randn(6).astype(DTYPE)
    x = Tensor(x_np, requires_grad=True)
    out = tanh(x)
    out.sum().backward()

    xt = torch_from(x_np, requires_grad=True)
    outt = torch.tanh(xt)
    outt.sum().backward()

    assert np_allclose(out.data, outt.detach().numpy())
    assert np_allclose(x.grad, xt.grad.numpy())

def test_sigmoid_forward_and_grad():
    x_np = np.random.randn(6).astype(DTYPE)
    x = Tensor(x_np, requires_grad=True)
    out = sigmoid(x)
    out.sum().backward()

    xt = torch_from(x_np, requires_grad=True)
    outt = torch.sigmoid(xt)
    outt.sum().backward()

    assert np_allclose(out.data, outt.detach().numpy())
    assert np_allclose(x.grad, xt.grad.numpy())


def main():
    torch.set_default_dtype(TORCH_DTYPE)
    tests = [
        ("softmax_forward_and_grad", test_softmax_forward_and_grad),
        ("log_softmax_forward_and_grad", test_log_softmax_forward_and_grad),
        ("l1_loss_forward_and_grad", test_l1_loss_forward_and_grad),
        ("mse_loss_forward_and_grad", test_mse_loss_forward_and_grad),
        ("nll_loss_forward_and_grad", test_nll_loss_forward_and_grad),
        ("cross_entropy_forward_and_grad", test_cross_entropy_forward_and_grad),
        ("relu_forward_and_grad", test_relu_forward_and_grad),
        ("tanh_forward_and_grad", test_tanh_forward_and_grad),
        ("sigmoid_forward_and_grad", test_sigmoid_forward_and_grad),
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