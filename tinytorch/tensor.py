import numpy as np


class Tensor:
    """
    A miniaturized version of torch.tensor
        Parameters:
            data: list, np.ndarray, int, float
            dtype: np.dtype
            requires_grad: bool
        Attributes:
            data: np.ndarray
            shape: list
            dtype: np.dtype
            requires_grad: bool
            grad: np.ndarray
            _grad_fn: function
    """

    def __init__(self, data, _children=(), requires_grad=False, dtype=np.float64):
        if isinstance(data, list):
            self.data = np.array(data, dtype=dtype)
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, (int, float)):
            self.data = np.array(data)
        else:
            raise TypeError("Invalid data type for Tensor")
        self.shape = list(self.data.shape)
        self.dtype = self.data.dtype
        self._prev = set(_children)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data)  # Tensor(, (), False, np.float64)
        self._grad_fn = lambda: None

    def __repr__(self):
        return f"Tensor({self.data})"

    def __add__(self, other):
        out = Tensor(0)
        if isinstance(other, list):
            raise TypeError("Invalid data type for Tensor")
        elif isinstance(other, np.ndarray):
            out = Tensor(self.data + other, (self,), requires_grad=self.requires_grad)
        elif isinstance(other, (int, float)):
            out = Tensor(self.data + other, (self,), requires_grad=self.requires_grad)
        elif isinstance(other, Tensor):
            out = Tensor(
                self.data + other.data,
                (self, other),
                requires_grad=self.requires_grad or other.requires_grad,
            )

        def _grad_fn():
            if isinstance(other, (int, float, np.ndarray)):
                if self.requires_grad:
                    self.grad += 1.0 * out.grad
            else:
                if self.requires_grad or other.requires_grad:
                    if self.requires_grad:
                        self.grad += 1.0 * out.grad
                    if other.requires_grad:
                        other.grad += 1.0 * out.grad

        out._grad_fn = _grad_fn
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        out = Tensor(0)
        if isinstance(other, list):
            raise TypeError("Invalid data type for Tensor")
        elif isinstance(other, np.ndarray):
            out = Tensor(self.data * other, (self,), requires_grad=self.requires_grad)
        elif isinstance(other, (int, float)):
            out = Tensor(self.data * other, (self,), requires_grad=self.requires_grad)
        elif isinstance(other, Tensor):
            out = Tensor(
                self.data * other.data,
                (self, other),
                requires_grad=self.requires_grad or other.requires_grad,
            )

        def _grad_fn():
            if isinstance(other, (int, float, np.ndarray)):
                if self.requires_grad:
                    self.grad += other * out.grad
            else:
                if self.requires_grad or other.requires_grad:
                    if self.requires_grad:
                        self.grad += other.data * out.grad
                    if other.requires_grad:
                        other.grad += self.data * out.grad

        out._grad_fn = _grad_fn

        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, other):
        out = Tensor(0)
        if isinstance(other, (int, float)):
            out = Tensor(self.data**other, (self,))
        elif isinstance(other, Tensor):
            if self.shape != other.shape:
                raise TypeError("Invalid shape for Tensor")
            out = Tensor(self.data**other.data, (self, other))
        else:
            raise TypeError("Invalid data type for Tensor")

        if isinstance(other, (int, float)):
            if self.requires_grad:
                self.grad += (other * (self.data ** (other - 1))) * out.grad
        else:
            if self.requires_grad or other.requires_grad:
                if self.requires_grad:
                    self.grad += (
                        other.data * (self.data ** (other - 1).data)
                    ) * out.grad
                if other.requires_grad:
                    other.grad += (
                        (self.data**other.data) * np.log(self.data) * out.grad
                    )

    def __truediv__(self, other):
        return self * (other**-1)

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), requires_grad=self.requires_grad)

        def _grad_fn():
            self.grad += out.data * out.grad

        out._grad_fn = _grad_fn

        return out

    def view(self, shape):
        self.data = self.data.reshape(shape)
        return self

    def backward(self):
        topo_order = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo_order.append(t)

        build_topo(self)

        self.grad = np.ones_like(self.data)
        for t in reversed(topo_order):
            t._grad_fn()


def arange(start=0, end=None, step=1, requires_grad=False, dtype=np.float64):
    """
        Similar to np.arange or torch.arange
    """
    if end is None:
        end = start
        start = 0
    return Tensor(np.arange(start, end, step), requires_grad=requires_grad)
