import numpy as np

#######
# TENSOR CORE
#######

# NOTE: VERY CRUCIAL FIX (CREDITS: gemini-2.5-pro)
def _unbroadcast(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Sums a gradient along axes that were broadcasted in the forward pass.
    This is the reverse of numpy's broadcasting operation.
    """
    if target_shape == grad.shape:
        return grad
    
    # 1. Sum over extra leading dimensions created by broadcasting.
    # e.g., grad shape (2,3), target shape (3,) -> sum over axis 0.
    n_extra_dims = grad.ndim - len(target_shape)
    if n_extra_dims > 0:
        grad = grad.sum(axis=tuple(range(n_extra_dims)))

    # 2. Sum over dimensions that were 1 in the target shape.
    # e.g., grad shape (2,5), target shape (1,5) -> sum over axis 0.
    axes_to_sum = []
    for i, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[i] > 1:
            axes_to_sum.append(i)
    
    if axes_to_sum:
        grad = grad.sum(axis=tuple(axes_to_sum), keepdims=True)
    
    return grad.reshape(target_shape)

class Tensor:
    """
    A miniaturized version of torch.tensor
        Parameters:
            data: list | np.ndarray | int | float | bool
            dtype: np.dtype | optional
            requires_grad: bool, default False
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
            self.data = np.asarray(data, dtype=dtype)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(dtype, copy=False)
        elif isinstance(data, (int, float, np.number, bool)):
            self.data = np.asarray(data, dtype=dtype)
        else:
            raise TypeError(f"Unsupported data type {type(data)}")
        
        # METADATA
        self.shape = list(self.data.shape)
        self.dtype = self.data.dtype
        self._prev = set(_children)
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = lambda: None
        
        # for reduction operations like max, min, maximum
        self.indices = None
        
        
    def _init_grad(self):
        # allocate .grad if missing lazily
        if self.grad is None:
            self.grad = np.zeros_like(self.data, dtype=self.dtype)
            
    def __hash__(self):
        return id(self)

    @property
    def T(self):
        out = Tensor(self.data.T, (self,), requires_grad=self.requires_grad)
        def _grad_fn():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad.T
        out._grad_fn = _grad_fn
        return out


    def __repr__(self):
        if self.indices is not None:
            return f"Tensor({self.data!r}, requires_grad={self.requires_grad}, indices={self.indices})"
        else:
            return f"Tensor({self.data!r}, requires_grad={self.requires_grad})"
    
    # ELEMENTAL OPERATIONS
    
    @staticmethod
    def _data(x):
        return x.data if isinstance(x, Tensor) else x

    def __add__(self, other):
        other_tensor = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data + other_tensor.data
        out = Tensor(out_data, (self, other_tensor), requires_grad=self.requires_grad or other_tensor.requires_grad)

        def _grad_fn():
            if self.requires_grad:
                self._init_grad()
                self.grad += _unbroadcast(out.grad, self.shape)
            if other_tensor.requires_grad:
                other_tensor._init_grad()
                other_tensor.grad += _unbroadcast(out.grad, other_tensor.shape)
        out._grad_fn = _grad_fn
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other_tensor = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data * other_tensor.data
        out = Tensor(out_data, (self, other_tensor), requires_grad=self.requires_grad or other_tensor.requires_grad)

        def _grad_fn():
            if self.requires_grad:
                self._init_grad()
                grad = other_tensor.data * out.grad
                self.grad += _unbroadcast(grad, self.shape)
            if other_tensor.requires_grad:
                other_tensor._init_grad()
                grad = self.data * out.grad
                other_tensor.grad += _unbroadcast(grad, other_tensor.shape)
        out._grad_fn = _grad_fn
        return out
    
    __rmul__ = __mul__
    
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return  (-self) + other
    
    def __pow__(self, exponent):
        exp_tensor = exponent if isinstance(exponent, Tensor) else Tensor(exponent)
        out_data = self.data ** exp_tensor.data
        out = Tensor(out_data, (self, exp_tensor), requires_grad=self.requires_grad or exp_tensor.requires_grad)

        def _grad_fn():
            if self.requires_grad:
                self._init_grad()
                grad = (exp_tensor.data * self.data**(exp_tensor.data - 1.0)) * out.grad
                self.grad += _unbroadcast(grad, self.shape)
            if exp_tensor.requires_grad:
                exp_tensor._init_grad()
                grad = (out_data * np.log(self.data)) * out.grad
                exp_tensor.grad += _unbroadcast(grad, exp_tensor.shape)
        out._grad_fn = _grad_fn
        return out
    
    def __truediv__(self, other):
        return self * (other**-1.0)
    
    # COMPARISONS
    
    def _cmp(self, other, op):
        return Tensor(op(self.data, self._data(other)), dtype=np.bool_)

    def __gt__(self, o):  return self._cmp(o, np.greater)
    def __lt__(self, o):  return self._cmp(o, np.less)
    def __ge__(self, o):  return self._cmp(o, np.greater_equal)
    def __le__(self, o):  return self._cmp(o, np.less_equal)
    def __eq__(self, other):
        return Tensor(self.data == self._data(other), dtype=np.bool_)
    def __ne__(self, other):
        return Tensor(self.data != self._data(other), dtype=np.bool_)
    
    # MATRIX MULTIPLICATION

    def __matmul__(self, other):
        other_tensor = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data @ other_tensor.data
        out = Tensor(out_data, (self, other_tensor), requires_grad=self.requires_grad or other_tensor.requires_grad)
        
        def _grad_fn():
            if self.requires_grad:
                self._init_grad()
                grad = out.grad @ other_tensor.data.swapaxes(-1, -2)
                self.grad += _unbroadcast(grad, self.shape)
            if other_tensor.requires_grad:
                other_tensor._init_grad()
                grad = self.data.swapaxes(-1, -2) @ out.grad
                other_tensor.grad += _unbroadcast(grad, other_tensor.shape)
        out._grad_fn = _grad_fn
        return out
    
    # REDUCTIONS
    def sum(self, dim=None, keepdim=True):
        out = Tensor(np.sum(self.data, axis=dim, keepdims=keepdim), (self,), requires_grad=self.requires_grad)

        def _grad_fn():
            if self.requires_grad:
                self._init_grad()
                grad = out.grad
                
                # When a dimension is summed over and keepdim=False, its rank is reduced.
                # We must re-insert that dimension (with size 1) to broadcast the
                # gradient back to the original shape.
                if dim is not None and not keepdim:
                    grad = np.expand_dims(grad, axis=dim)
                
                self.grad += np.broadcast_to(grad, self.shape)
        out._grad_fn = _grad_fn
        return out
    
    def mean(self, dim=None, keepdim=False):
        total = self.sum(dim=dim, keepdim=keepdim)
        n = self.data.size / total.data.size if dim is None else self.shape[dim]
        return total / n
    
    def max(self, dim=None, keepdim=False):
        val = np.max(self.data, axis=dim, keepdims=keepdim)
        indices = np.argmax(self.data, axis=dim, keepdims=keepdim)
        out = Tensor(val, (self,), requires_grad=self.requires_grad)
        out.indices = indices
        
        def _grad_fn():
            if self.requires_grad:
                self._init_grad()
                grad_input = np.zeros_like(self.data, dtype=self.dtype)
                
                grad_out = out.grad
                idx = out.indices
                
                if dim is None:
                    # Global max case: indices is a scalar, need to handle differently
                    grad_input.flat[idx] = grad_out
                else:
                    # Specific dimension case
                    if not keepdim:
                        grad_out = np.expand_dims(grad_out, axis=dim)
                        idx = np.expand_dims(idx, axis=dim)
                        
                    np.put_along_axis(grad_input, idx, grad_out, axis=dim)
                    
                self.grad += grad_input
                
        out._grad_fn = _grad_fn
        return out
    
    def min(self, dim=None, keepdim=False):
        val = np.min(self.data, axis=dim, keepdims=keepdim)
        indices = np.argmin(self.data, axis=dim, keepdims=keepdim)
        out = Tensor(val, (self,), requires_grad=self.requires_grad)
        out.indices = indices

        def _grad_fn():
            if self.requires_grad:
                self._init_grad()
                grad_input = np.zeros_like(self.data, dtype=self.dtype)
                
                grad_out = out.grad
                idx = out.indices
                
                if dim is None:
                    grad_input.flat[idx] = grad_out
                else:
                    if not keepdim:
                        grad_out = np.expand_dims(grad_out, axis=dim)
                        idx = np.expand_dims(idx, axis=dim)
                        
                    np.put_along_axis(grad_input, idx, grad_out, axis=dim)
                    
                self.grad += grad_input
                
        out._grad_fn = _grad_fn
        return out

        
    # ELEMENT WISE UNARY

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), requires_grad=self.requires_grad)
        def _grad_fn():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.data * out.grad
        out._grad_fn = _grad_fn
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), requires_grad=self.requires_grad)
        def _grad_fn():
            if self.requires_grad:
                self._init_grad()
                self.grad += (1 / self.data) * out.grad
        out._grad_fn = _grad_fn
        return out

    def abs(self):
        out = Tensor(np.abs(self.data), (self,), requires_grad=self.requires_grad)
        def _grad_fn():
            if self.requires_grad:
                self._init_grad()
                self.grad += np.sign(self.data) * out.grad
        out._grad_fn = _grad_fn
        return out
    
    # CREDITS: gemini-pro-2.5
    def maximum(self, other):
        other_tensor = other if isinstance(other, Tensor) else Tensor(other)
        out_data = np.maximum(self.data, other_tensor.data)
        out = Tensor(out_data, (self, other_tensor), requires_grad=self.requires_grad or other_tensor.requires_grad)

        def _grad_fn():
            if self.requires_grad:
                self._init_grad()
                # Grad is 1 where self > other, 0.5 where self == other, 0 where self < other
                mask = (self.data >= other_tensor.data).astype(self.dtype)
                mask[self.data == other_tensor.data] = 0.5
                self.grad += _unbroadcast(mask * out.grad, self.shape)
            if other_tensor.requires_grad:
                other_tensor._init_grad()
                # Grad is 1 where other > self, 0.5 where self == other, 0 where other < self
                mask = (other_tensor.data >= self.data).astype(other_tensor.dtype)
                mask[other_tensor.data == self.data] = 0.5
                other_tensor.grad += _unbroadcast(mask * out.grad, other_tensor.shape)

        out._grad_fn = _grad_fn
        return out
    
    # SHAPE OPERATIONS
    
    def _view_like(self, new_data):
        out = Tensor(new_data, (self,), requires_grad=self.requires_grad)
        def _grad_fn():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad.reshape(self.shape)
        out._grad_fn = _grad_fn
        return out

    def view(self, shape): return self._view_like(self.data.reshape(shape))
    def reshape(self, shape): return self._view_like(self.data.reshape(shape))
    def flatten(self): return self._view_like(self.data.flatten())
    def unsqueeze(self, dim): return self._view_like(np.expand_dims(self.data, dim))
    def squeeze(self, dim=None): return self._view_like(np.squeeze(self.data, axis=dim))
    
    # INDEXING
    
    def __getitem__(self, idx):
        
        if isinstance(idx, Tensor):
            raise NotImplementedError("support for idx to be Tensor object not present yet, please pass either a numpy array or a python list")
        
        out = Tensor(self.data[idx], (self,), requires_grad=self.requires_grad)
        def _grad_fn():
            if self.requires_grad:
                self._init_grad()
                # Create a zero grad and add the output grad in the correct slice
                grad_slice = np.zeros_like(self.data, dtype=self.dtype)
                grad_slice[idx] = out.grad
                self.grad += grad_slice
        out._grad_fn = _grad_fn
        return out
    
    # AUTOGRAD ENGINE

    def zero_grad(self):
        self.grad = np.zeros_like(self.data, dtype=self.dtype)

    def backward(self, gradient=None):
        
        if self.data.size != 1 and gradient is None:
            raise RuntimeError(" If the tensor is non-scalar (i.e. its data has more than one element) and requires gradient, the function additionally requires specifying a `gradient`")
        
        topo = []
        seen = set()
        
        def build(t):
            if t not in seen:
                seen.add(t)
                for child in t._prev:
                    if isinstance(child, Tensor):
                        build(child)
                topo.append(t)
        
        build(self)
        
        self._init_grad() # initialize the gradient for the final tensor
        
        if gradient is None:
            self.grad.fill(1.0)
        else:
            if isinstance(gradient, Tensor):
                gradient = gradient.data
            if not isinstance(gradient, np.ndarray):
                gradient = np.array(gradient, dtype=self.dtype)
            if gradient.shape != self.data.shape:
                raise RuntimeError(f"provided gradient shape: {gradient.shape} does not match tensot shape {self.data.shape}")
            
            self.grad[...] = gradient
        
        for node in reversed(topo):
            node._grad_fn()


# HELPER FUNCTIONS

def arange(start=0, end=None, step=1, requires_grad=False, dtype=np.int64):
    """
    Similar to np.arange or torch.arange
    """
    if end is None:
        end = start
        start = 0
    return Tensor(np.arange(start, end, step, dtype=dtype), requires_grad=requires_grad)


def broadcast_tensors(t1, t2):
    try:
        d1, d2 = np.broadcast_arrays(t1.data, t2.data)
        return (Tensor(d1, requires_grad=t1.requires_grad),
                Tensor(d2, requires_grad=t2.requires_grad))
    except ValueError as e:
        raise ValueError(f"Incompatible shapes: {e}") from None


def randn(shape: tuple, requires_grad=False):
    """
    Returns a tensor with normally distributed values
    """
    return Tensor(np.random.randn(*shape) * 0.1, requires_grad=requires_grad)

# CREDITS: (claude-4.0-sonnet)
def where(condition, x=0, y=1):
    cond = condition if isinstance(condition, Tensor) else Tensor(condition)
    x_tensor = x if isinstance(x, Tensor) else Tensor(x)
    y_tensor = y if isinstance(y, Tensor) else Tensor(y)
    
    # Compute the output
    out_data = np.where(cond.data, x_tensor.data, y_tensor.data)
    out = Tensor(out_data, (x_tensor, y_tensor), 
                 requires_grad=x_tensor.requires_grad or y_tensor.requires_grad)
    
    def _grad_fn():
        if x_tensor.requires_grad:
            x_tensor._init_grad()
            # Gradient flows to x where condition is True
            x_mask = cond.data.astype(x_tensor.dtype)
            x_grad = x_mask * out.grad
            x_tensor.grad += _unbroadcast(x_grad, x_tensor.shape)
            
        if y_tensor.requires_grad:
            y_tensor._init_grad()
            # Gradient flows to y where condition is False
            # Use logical_not instead of bitwise invert
            y_mask = np.logical_not(cond.data).astype(y_tensor.dtype)
            y_grad = y_mask * out.grad
            y_tensor.grad += _unbroadcast(y_grad, y_tensor.shape)
    
    out._grad_fn = _grad_fn
    return out
