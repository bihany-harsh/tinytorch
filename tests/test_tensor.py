# CREDITS: (gpt-5)

import numpy as np
import torch
import traceback
import sys

from tinytorch.tensor import Tensor, broadcast_tensors, where

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

def test_backward_scalar_matches_torch():
    x_np = np.array(3.0, dtype=DTYPE)
    x = Tensor(x_np, requires_grad=True)
    y = x * x + 2.0
    y.backward()
    # torch
    xt = torch_from(x_np, requires_grad=True)
    yt = xt * xt + 2.0
    yt.backward()
    assert np_allclose(x.data, xt.detach().numpy())
    assert np_allclose(x.grad, xt.grad.numpy())

def test_backward_non_scalar_requires_gradient_argument_error():
    a_np = np.arange(6., dtype=DTYPE).reshape(2, 3)
    a = Tensor(a_np, requires_grad=True)
    z = a * 2.0
    # our Tensor should require a gradient argument for non-scalar backward
    caught = False
    try:
        z.backward()
    except RuntimeError:
        caught = True
    assert caught, "Expected RuntimeError for non-scalar backward without gradient"

    # torch behavior: same requirement
    at = torch_from(a_np, requires_grad=True)
    zt = at * 2.0
    caught_torch = False
    try:
        zt.backward()
    except RuntimeError:
        caught_torch = True
    assert caught_torch, "Expected RuntimeError in torch for non-scalar backward without gradient"

def test_broadcast_add_forward_and_grad():
    a_np = np.arange(6., dtype=DTYPE).reshape(2, 3)
    b_np = np.array([1., 2., 3.], dtype=DTYPE)
    a = Tensor(a_np, requires_grad=True)
    b = Tensor(b_np, requires_grad=True)
    
    out = (a + b).sum(keepdim=False)
    out.backward()
    # torch
    at = torch_from(a_np, requires_grad=True)
    bt = torch_from(b_np, requires_grad=True)
    outt = (at + bt).sum()
    outt.backward()
    assert np_allclose((a + b).data, (at + bt).detach().numpy())
    assert np_allclose(a.grad, at.grad.numpy())
    assert np_allclose(b.grad, bt.grad.numpy())

def test_matmul_forward_and_grad():
    A_np = np.arange(6., dtype=DTYPE).reshape(2, 3)
    B_np = np.arange(12., dtype=DTYPE).reshape(3, 4) / 10.0
    A = Tensor(A_np, requires_grad=True)
    B = Tensor(B_np, requires_grad=True)
    out = (A @ B).sum(keepdim=False)
    out.backward()
    # torch
    At = torch_from(A_np, requires_grad=True)
    Bt = torch_from(B_np, requires_grad=True)
    outt = (At @ Bt).sum()
    outt.backward()
    assert np_allclose((A @ B).data, (At @ Bt).detach().numpy())
    assert np_allclose(A.grad, At.grad.numpy())
    assert np_allclose(B.grad, Bt.grad.numpy())

def test_pow_scalar_exponent_grad():
    x_np = np.array([0.5, 1.5, 2.0], dtype=DTYPE)
    x = Tensor(x_np, requires_grad=True)
    y = (x ** 3.0).sum(keepdim=False)
    y.backward()
    # torch
    xt = torch_from(x_np, requires_grad=True)
    yt = (xt ** 3.0).sum()
    yt.backward()
    assert np_allclose((x ** 3.0).data, (xt ** 3.0).detach().numpy())
    assert np_allclose(x.grad, xt.grad.numpy())

def test_pow_tensor_exponent_grad():
    base_np = np.array([1.2, 2.3, 3.4], dtype=DTYPE)
    exp_np = np.array(2.5, dtype=DTYPE)
    base = Tensor(base_np, requires_grad=True)
    exp = Tensor(exp_np, requires_grad=True)
    y = (base ** exp).sum(keepdim=False)
    y.backward()
    # torch
    baset = torch_from(base_np, requires_grad=True)
    expt = torch_from(exp_np, requires_grad=True)
    yt = (baset ** expt).sum()
    yt.backward()
    assert np_allclose((base ** exp).data, (baset ** expt).detach().numpy())
    assert np_allclose(base.grad, baset.grad.numpy())
    assert np_allclose(exp.grad, expt.grad.numpy())

def test_division_grad_and_broadcast():
    x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=DTYPE)
    y_np = np.array(2.0, dtype=DTYPE)
    x = Tensor(x_np, requires_grad=True)
    y = Tensor(y_np, requires_grad=True)
    out = (x / y).sum(keepdim=False)
    out.backward()
    # torch
    xt = torch_from(x_np, requires_grad=True)
    yt = torch_from(y_np, requires_grad=True)
    outt = (xt / yt).sum()
    outt.backward()
    assert np_allclose((x / y).data, (xt / yt).detach().numpy())
    assert np_allclose(x.grad, xt.grad.numpy())
    assert np_allclose(y.grad, yt.grad.numpy())

def test_sum_mean_keepdim_and_grads():
    X_np = np.arange(1., 13., dtype=DTYPE).reshape(3, 4)
    X = Tensor(X_np, requires_grad=True)
    s1 = X.sum(dim=1, keepdim=True)
    s2 = X.sum(dim=1, keepdim=False)
    m1 = X.mean(dim=0, keepdim=True)
    m2 = X.mean(dim=0, keepdim=False)
    g = s1.sum() + s2.sum() + m1.sum() + m2.sum()
    
    g.backward()

    Xt = torch_from(X_np, requires_grad=True)
    ts1 = Xt.sum(dim=1, keepdim=True)
    ts2 = Xt.sum(dim=1, keepdim=False)
    tm1 = Xt.mean(dim=0, keepdim=True)
    tm2 = Xt.mean(dim=0, keepdim=False)
    gt = ts1.sum() + ts2.sum() + tm1.sum() + tm2.sum()
    gt.backward()

    assert s1.data.shape == ts1.shape
    assert s2.data.shape == ts2.shape
    assert m1.data.shape == tm1.shape
    assert m2.data.shape == tm2.shape
    assert np_allclose(s1.data, ts1.detach().numpy())
    assert np_allclose(s2.data, ts2.detach().numpy())
    assert np_allclose(m1.data, tm1.detach().numpy())
    assert np_allclose(m2.data, tm2.detach().numpy())
    assert np_allclose(X.grad, Xt.grad.numpy())
    
def test_max_reduction_grad():
    X_np = np.array([[-1, 1, 5], [2, -6, 4]], dtype=DTYPE)
    # Test with dim and keepdim=True
    X = Tensor(X_np, requires_grad=True)
    out_max1 = X.max(dim=1, keepdim=True)
    out_max1.sum().backward()
    Xt = torch_from(X_np, requires_grad=True)
    out_maxt1 = Xt.max(dim=1, keepdim=True).values
    out_maxt1.sum().backward()
    assert np_allclose(out_max1.data, out_maxt1.detach().numpy())
    assert np_allclose(X.grad, Xt.grad.numpy())
    
    # Test with dim and keepdim=False
    X.zero_grad()
    Xt.grad.zero_()
    out_max2 = X.max(dim=0, keepdim=False)
    out_max2.sum().backward()
    out_maxt2 = Xt.max(dim=0, keepdim=False).values
    out_maxt2.sum().backward()
    assert np_allclose(out_max2.data, out_maxt2.detach().numpy())
    assert np_allclose(X.grad, Xt.grad.numpy())
    
    # Test on whole tensor
    X.zero_grad()
    Xt.grad.zero_()
    out_max3 = X.max()
    out_max3.backward()
    out_maxt3 = Xt.max()
    out_maxt3.backward()
    assert np_allclose(out_max3.data, out_maxt3.detach().numpy())
    assert np_allclose(X.grad, Xt.grad.numpy())
    
def test_min_reduction_grad():
    X_np = np.array([[-1, 1, 5], [2, -6, 4]], dtype=DTYPE)
    # Test with dim and keepdim=True
    X = Tensor(X_np, requires_grad=True)
    out_min1 = X.min(dim=1, keepdim=True)
    out_min1.sum().backward()
    Xt = torch_from(X_np, requires_grad=True)
    out_mint1 = Xt.min(dim=1, keepdim=True).values
    out_mint1.sum().backward()
    assert np_allclose(out_min1.data, out_mint1.detach().numpy())
    assert np_allclose(X.grad, Xt.grad.numpy())
    
    # Test with dim and keepdim=False
    X.zero_grad()
    Xt.grad.zero_()
    out_min2 = X.min(dim=0, keepdim=False)
    out_min2.sum().backward()
    out_mint2 = Xt.min(dim=0, keepdim=False).values
    out_mint2.sum().backward()
    assert np_allclose(out_min2.data, out_mint2.detach().numpy())
    assert np_allclose(X.grad, Xt.grad.numpy())
    
    # Test on whole tensor
    X.zero_grad()
    Xt.grad.zero_()
    out_min3 = X.min()
    out_min3.backward()
    out_mint3 = Xt.min()
    out_mint3.backward()
    assert np_allclose(out_min3.data, out_mint3.detach().numpy())
    assert np_allclose(X.grad, Xt.grad.numpy())
    
def test_maximum_elementwise_grad():
    a_np = np.array([1, 5, 3], dtype=DTYPE)
    b_np = np.array([4, 2, 3], dtype=DTYPE)
    a = Tensor(a_np, requires_grad=True)
    b = Tensor(b_np, requires_grad=True)
    out = a.maximum(b)
    out.sum().backward()

    at = torch_from(a_np, requires_grad=True)
    bt = torch_from(b_np, requires_grad=True)
    outt = torch.maximum(at, bt)
    outt.sum().backward()

    assert np_allclose(out.data, outt.detach().numpy())
    assert np_allclose(a.grad, at.grad.numpy())
    assert np_allclose(b.grad, bt.grad.numpy())

def test_indexing_getitem_grad():
    X_np = np.arange(1., 7., dtype=DTYPE).reshape(2, 3)
    X = Tensor(X_np, requires_grad=True)
    y = (X[0, :] * np.array([1.0, 2.0, 3.0], dtype=DTYPE)).sum()
    y.backward()
    # torch
    Xt = torch_from(X_np, requires_grad=True)
    yt = (Xt[0, :] * torch.tensor([1.0, 2.0, 3.0], dtype=TORCH_DTYPE)).sum()
    yt.backward()
    assert np_allclose(y.data, yt.detach().numpy())
    assert np_allclose(X.grad, Xt.grad.numpy())

def test_broadcast_tensors_helper_matches_torch():
    a_np = np.ones((2, 1, 3), dtype=DTYPE)
    b_np = np.ones((1, 3), dtype=DTYPE)
    a = Tensor(a_np, requires_grad=False)
    b = Tensor(b_np, requires_grad=False)
    ba, bb = broadcast_tensors(a, b)
    # torch
    at = torch_from(a_np)
    bt = torch_from(b_np)
    ta, tb = torch.broadcast_tensors(at, bt)
    assert ba.data.shape == ta.shape
    assert bb.data.shape == tb.shape

def test_where_forward_and_grad():
    cond = np.array([[True, False], [False, True]])
    x_np = np.array([[1., 2.], [3., 4.]], dtype=DTYPE)
    y_np = np.array([[10., 20.], [30., 40.]], dtype=DTYPE)
    
    x = Tensor(x_np, requires_grad=True)
    y = Tensor(y_np, requires_grad=True)
    out = where(cond, x, y)
    out.sum().backward()
    
    xt = torch_from(x_np, requires_grad=True)
    yt = torch_from(y_np, requires_grad=True)
    condt = torch.tensor(cond)
    outt = torch.where(condt, xt, yt)
    outt.sum().backward()

    assert np_allclose(out.data, outt.detach().numpy())
    assert np_allclose(x.grad, xt.grad.numpy())
    assert np_allclose(y.grad, yt.grad.numpy())


def main():
    torch.set_default_dtype(TORCH_DTYPE)
    tests = [
        ("backward_scalar_matches_torch", test_backward_scalar_matches_torch),
        ("backward_non_scalar_requires_gradient_argument_error", test_backward_non_scalar_requires_gradient_argument_error),
        ("broadcast_add_forward_and_grad", test_broadcast_add_forward_and_grad),
        ("matmul_forward_and_grad", test_matmul_forward_and_grad),
        ("pow_scalar_exponent_grad", test_pow_scalar_exponent_grad),
        ("pow_tensor_exponent_grad", test_pow_tensor_exponent_grad),
        ("division_grad_and_broadcast", test_division_grad_and_broadcast),
        ("sum_mean_keepdim_and_grads", test_sum_mean_keepdim_and_grads),
        ("max_reduction_grad", test_max_reduction_grad),
        ("min_reduction_grad", test_min_reduction_grad),
        ("maximum_elementwise_grad", test_maximum_elementwise_grad),
        ("indexing_getitem_grad", test_indexing_getitem_grad),
        ("broadcast_tensors_helper_matches_torch", test_broadcast_tensors_helper_matches_torch),
        ("where_forward_and_grad", test_where_forward_and_grad),
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
