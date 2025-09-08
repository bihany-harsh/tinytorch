# CREDITS: gpt-5
import numpy as np
import torch
import traceback
import sys

from tinytorch.tensor import Tensor, randn
from tinytorch.optim import SGD, RMSprop, Adam

DTYPE = np.float64
TORCH_DTYPE = torch.float64

def np_allclose(a, b, rtol=1e-6, atol=1e-8):
    return np.allclose(np.asarray(a, dtype=DTYPE), np.asarray(b, dtype=DTYPE), rtol=rtol, atol=atol)

def run_test(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        traceback.print_exc()
        return False


##############################
# SGD TESTS
##############################

def test_sgd_basic_update_matches_torch():
    w_np = np.array([1.0, -2.0, 3.0], dtype=DTYPE)
    g_np = np.array([0.1, -0.2, 0.3], dtype=DTYPE)

    w = Tensor(w_np.copy(), requires_grad=True)
    w.grad = g_np.copy()
    opt = SGD([w], lr=0.1)
    opt.step()

    # torch
    wt = torch.tensor(w_np, dtype=TORCH_DTYPE, requires_grad=True)
    wt.grad = torch.tensor(g_np, dtype=TORCH_DTYPE)
    optt = torch.optim.SGD([wt], lr=0.1)
    optt.step()

    assert np_allclose(w.data, wt.detach().numpy())

def test_sgd_momentum_and_nesterov_match_torch():
    w_np = np.array([0.5, -1.5], dtype=DTYPE)
    g_np = np.array([0.3, -0.7], dtype=DTYPE)

    for nesterov in [False, True]:
        w = Tensor(w_np.copy(), requires_grad=True)
        w.grad = g_np.copy()
        opt = SGD([w], lr=0.05, momentum=0.9, nesterov=nesterov)
        for _ in range(5):
            opt.step()
            w.grad = g_np.copy()  # constant grad

        # torch
        wt = torch.tensor(w_np, dtype=TORCH_DTYPE, requires_grad=True)
        wt.grad = torch.tensor(g_np, dtype=TORCH_DTYPE)
        optt = torch.optim.SGD([wt], lr=0.05, momentum=0.9, nesterov=nesterov)
        for _ in range(5):
            optt.step()
            wt.grad = torch.tensor(g_np, dtype=TORCH_DTYPE)

        assert np_allclose(w.data, wt.detach().numpy())

def test_sgd_weight_decay_and_maximize():
    w_np = np.array([1.0, -1.0], dtype=DTYPE)
    g_np = np.array([0.2, -0.2], dtype=DTYPE)

    w1 = Tensor(w_np.copy(), requires_grad=True)
    w1.grad = g_np.copy()
    opt1 = SGD([w1], lr=0.1, weight_decay=0.01, maximize=False)
    opt1.step()

    w2 = Tensor(w_np.copy(), requires_grad=True)
    w2.grad = g_np.copy()
    opt2 = SGD([w2], lr=0.1, weight_decay=0.01, maximize=True)
    opt2.step()

    # torch
    wt1 = torch.tensor(w_np, dtype=TORCH_DTYPE, requires_grad=True)
    wt1.grad = torch.tensor(g_np, dtype=TORCH_DTYPE)
    optt1 = torch.optim.SGD([wt1], lr=0.1, weight_decay=0.01)
    optt1.step()

    wt2 = torch.tensor(w_np, dtype=TORCH_DTYPE, requires_grad=True)
    wt2.grad = torch.tensor(g_np, dtype=TORCH_DTYPE)
    optt2 = torch.optim.SGD([wt2], lr=0.1, weight_decay=0.01)
    optt2.step(closure=None)  # default minimize
    wt2.data = -wt2.data  # crude simulate maximize (direction flip)

    assert np_allclose(w1.data, wt1.detach().numpy())
    assert not np_allclose(w1.data, w2.data)  # maximize should differ


##############################
# RMSprop TESTS
##############################

def test_rmsprop_basic_matches_torch():
    w_np = np.array([1.0, 2.0], dtype=DTYPE)
    g_np = np.array([0.5, -0.3], dtype=DTYPE)

    w = Tensor(w_np.copy(), requires_grad=True)
    w.grad = g_np.copy()
    opt = RMSprop([w], lr=0.01, alpha=0.9)
    for _ in range(5):
        opt.step()
        w.grad = g_np.copy()

    # torch
    wt = torch.tensor(w_np, dtype=TORCH_DTYPE, requires_grad=True)
    wt.grad = torch.tensor(g_np, dtype=TORCH_DTYPE)
    optt = torch.optim.RMSprop([wt], lr=0.01, alpha=0.9)
    for _ in range(5):
        optt.step()
        wt.grad = torch.tensor(g_np, dtype=TORCH_DTYPE)

    assert np_allclose(w.data, wt.detach().numpy())

def test_rmsprop_with_momentum_and_centered():
    w_np = np.array([-0.5, 0.5], dtype=DTYPE)
    g_np = np.array([0.2, -0.1], dtype=DTYPE)

    w = Tensor(w_np.copy(), requires_grad=True)
    w.grad = g_np.copy()
    opt = RMSprop([w], lr=0.01, alpha=0.95, momentum=0.9, centered=True)
    for _ in range(10):
        opt.step()
        w.grad = g_np.copy()

    assert np.isfinite(w.data).all()  # sanity: no NaNs or infs


##############################
# ADAM TESTS
##############################

def test_adam_basic_matches_torch():
    w_np = np.array([1.0, -1.0], dtype=DTYPE)
    g_np = np.array([0.1, -0.1], dtype=DTYPE)

    w = Tensor(w_np.copy(), requires_grad=True)
    w.grad = g_np.copy()
    opt = Adam([w], lr=0.01, betas=(0.9, 0.999))
    for _ in range(5):
        opt.step()
        w.grad = g_np.copy()

    wt = torch.tensor(w_np, dtype=TORCH_DTYPE, requires_grad=True)
    wt.grad = torch.tensor(g_np, dtype=TORCH_DTYPE)
    optt = torch.optim.Adam([wt], lr=0.01, betas=(0.9, 0.999))
    for _ in range(5):
        optt.step()
        wt.grad = torch.tensor(g_np, dtype=TORCH_DTYPE)

    assert np_allclose(w.data, wt.detach().numpy(), rtol=1e-5, atol=1e-7)


def test_zero_grad_resets_and_skips_frozen():
    w1 = Tensor(np.array([1.0]), requires_grad=True)
    w2 = Tensor(np.array([2.0]), requires_grad=False)  # frozen
    w1.grad = np.array([10.0])
    opt = Adam([w1, w2], lr=0.1)
    opt.zero_grad()
    assert np_allclose(w1.grad, 0.0)
    assert w2.grad is None  # frozen should be untouched


##############################
# MAIN
##############################

def main():
    torch.set_default_dtype(TORCH_DTYPE)
    tests = [
        ("sgd_basic_update_matches_torch", test_sgd_basic_update_matches_torch),
        ("sgd_momentum_and_nesterov_match_torch", test_sgd_momentum_and_nesterov_match_torch),
        ("sgd_weight_decay_and_maximize", test_sgd_weight_decay_and_maximize),
        ("rmsprop_basic_matches_torch", test_rmsprop_basic_matches_torch),
        ("rmsprop_with_momentum_and_centered", test_rmsprop_with_momentum_and_centered),
        ("adam_basic_matches_torch", test_adam_basic_matches_torch),
        ("zero_grad_resets_and_skips_frozen", test_zero_grad_resets_and_skips_frozen),
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
