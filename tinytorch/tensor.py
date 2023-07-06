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

    @property
    def T(self):
        out = Tensor(self.data.T, (self,), requires_grad=self.requires_grad)
        def _grad_fn():
            self.grad += out.grad.T
        out._grad_fn = _grad_fn

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

    def __matmul__(self, other):
        out = Tensor(0)
        if isinstance(other, (list, int, float)):
            raise TypeError(f"Invalid data type for @ {type(other)} and {type(self)}")
        elif isinstance(other, np.ndarray):
            if self.requires_grad:
                raise RuntimeError("Requires grad is not supported for np.ndarray")
            out = Tensor(self.data @ other, (self,), requires_grad=self.requires_grad)
        elif isinstance(other, Tensor):
            # 1. if both are 1D -> dot product
            if (len(self.shape)==1 and len(other.shape)==1):
                if self.shape[0] != other.shape[0]:
                    raise TypeError("Invalid shape for Tensor")
                out = Tensor(
                    self.data @ other.data,
                    (self, other),
                    requires_grad=self.requires_grad or other.requires_grad,
                )
            # 2. if both are 2D -> matrix multiplication
            elif (len(self.shape)==2 and len(other.shape)==2):
                if self.shape[1] != other.shape[0]:
                    raise TypeError("Invalid shape for Tensor")
                out = Tensor(
                    self.data @ other.data,
                    (self, other),
                    requires_grad=self.requires_grad or other.requires_grad,
                )
            # 3. if one is 1D and other is 2D -> matrix multiplication and result is 1D
            elif (len(self.shape)==1 and len(other.shape)==2):
                if self.shape[0] != other.shape[0]:
                    raise TypeError("Invalid shape for Tensor")
                out = Tensor(
                    self.data @ other.data,
                    (self, other),
                    requires_grad=self.requires_grad or other.requires_grad,
                )
            # 4. if one is 2D and other is 1D -> matrix multiplication and result is 1D
            elif (len(self.shape)==2 and len(other.shape)==1):
                if self.shape[1] != other.shape[0]:
                    raise TypeError("Invalid shape for Tensor")
                out = Tensor(
                    self.data @ other.data,
                    (self, other),
                    requires_grad=self.requires_grad or other.requires_grad,
                )
            # 5. broadcasting: (j x 1 x n x n) @ (k x n x n) -> (j x k x n x n) 
            # and (j×1×n×m) @ (k×m×p) -> (j×k×n×p)
            # if any of the dimensions are 1, then we can broadcast
            else:
                if self.shape[-1] != other.shape[-2]:
                    raise TypeError("Invalid shape for Tensor")
                # self_shape = self.shape
                # other_shape = other.shape
                # broadcast_shape = np.broadcast_shapes(self_shape, other_shape)
                # if broadcast_shape is None:
                #     raise TypeError("Incompatible shapes for broadcasting in matmul operation")
                
                out = Tensor(
                    np.matmul(self.data, other.data),
                    (self, other),
                    requires_grad=self.requires_grad or other.requires_grad,
                )
        else:
            raise TypeError("Invalid data type for Tensor")
        
        def _grad_fn():
            if isinstance(other, np.ndarray):
                if self.requires_grad:
                    self.grad += other.T * out.grad
            else:
                if self.requires_grad or other.requires_grad:
                    if self.requires_grad:
                        self.grad += out.grad @ other.data.T
                    if other.requires_grad:
                        other.grad += self.data.T @ out.grad

        out._grad_fn = _grad_fn
        return out


    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return  other + (-self)

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

        def _grad_fn():
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

        out._grad_fn = _grad_fn
        return out

    def __truediv__(self, other):
        return self * (other**-1)
    
    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self.data > other)
        elif isinstance(other, Tensor):
            return Tensor(self.data > other.data)
        else:
            raise TypeError("Invalid data type for comparison")
        
    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self.data < other)
        elif isinstance(other, Tensor):
            return Tensor(self.data < other.data)
        else:
            raise TypeError("Invalid data type for comparison")
        
    def __ge__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self.data >= other)
        elif isinstance(other, Tensor):
            return Tensor(self.data >= other.data)
        else:
            raise TypeError("Invalid data type for comparison")
        
    def __le__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self.data <= other)
        elif isinstance(other, Tensor):
            return Tensor(self.data <= other.data)
        else:
            raise TypeError("Invalid data type for comparison")
        
    # def __eq__(self, other):
    #     if isinstance(other, (int, float)):
    #         return Tensor((self.data == other).astype(int))
    #     elif isinstance(other, Tensor):
    #         return Tensor((self.data == other.data).astype(int))
    #     else:
    #         raise TypeError("Invalid data type for comparison")

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), requires_grad=self.requires_grad)

        def _grad_fn():
            self.grad += out.data * out.grad

        out._grad_fn = _grad_fn

        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), requires_grad=self.requires_grad)

        def _grad_fn():
            self.grad += (1 / self.data) * out.grad

        out._grad_fn = _grad_fn

        return out

    def mean(self, dim=None, keepdim=True):
        out = Tensor(
            np.mean(self.data, axis=dim, keepdims=keepdim),
            (self,),
            requires_grad=self.requires_grad,
        )

        def _grad_fn():
            self.grad += (1 / self.data.size) * out.grad

        out._grad_fn = _grad_fn
        return out

    def abs(self):
        out = Tensor(np.abs(self.data), (self,), requires_grad=self.requires_grad)

        def _grad_fn():
            self.grad += (self.data / np.abs(self.data)) * out.grad

        out._grad_fn = _grad_fn
        return out

    def sum(self, dim=None, keepdim=True):
        out = Tensor(
            np.sum(self.data, axis=dim, keepdims=keepdim),
            (self,),
            requires_grad=self.requires_grad,
        )

        def _grad_fn():
            self.grad += np.ones_like(self.data) * out.grad

        out._grad_fn = _grad_fn
        return out

    def view(self, shape):
        original_shape = self.data.shape
        self.data = self.data.reshape(shape)
        out = Tensor(self.data, (self,), requires_grad=self.requires_grad)

        def _grad_fn():
            self.grad += out.grad.reshape(original_shape)

        out._grad_fn = _grad_fn
        return out
    
    def reshape(self, shape):
        original_shape = self.data.shape
        self.data = self.data.reshape(shape)
        out = Tensor(self.data, (self,), requires_grad=self.requires_grad)

        def _grad_fn():
            self.grad += out.grad.reshape(original_shape)

        out._grad_fn = _grad_fn
        return out
    
    def flatten(self):
        original_shape = self.data.shape
        self.data = self.data.flatten()
        out = Tensor(self.data, (self,), requires_grad=self.requires_grad)

        def _grad_fn():
            self.grad += out.grad.reshape(original_shape)

        out._grad_fn = _grad_fn
        return out
    
    def unsqueeze(self, dim):
        self.data = np.expand_dims(self.data, dim)
        out = Tensor(self.data, (self,), requires_grad=self.requires_grad)

        def _grad_fn():
            self.grad += np.squeeze(out.grad, axis=dim)

        out._grad_fn = _grad_fn
        return out

    
    def one_hot(self, num_classes):
        data = np.zeros((self.data.size, num_classes))
        data[np.arange(self.data.size), self.data.flatten().astype(int)] = 1        
        out = Tensor(data.reshape((*self.data.shape, num_classes)), (self,), requires_grad=self.requires_grad)
        def _grad_fn():
            self.grad += out.reshape(self.shape) * out.grad
        out._grad_fn = _grad_fn
        return out
    
    def __getitem__(self, idx):
        i, j = idx
        if isinstance(i, int):
            i = slice(i, i+1)
        if isinstance(j, int):
            j = slice(j, j+1)
        if isinstance(i, Tensor):
            i = i.data.astype(int)
        if isinstance(j, Tensor):
            j = j.data.astype(int)

        data = self.data[i, j]
        if isinstance(data, np.ndarray):
            data = data.squeeze()
        out = Tensor(data, (self,), requires_grad=self.requires_grad)

        def _grad_fn():
            self.grad[i, j] += out.grad
        out._grad_fn = _grad_fn
        return out

    def __setitem__(self, idx, value):
        self.data[idx] = value

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def copy(self):
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

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


def arange(start=0, end=None, step=1, requires_grad=False, dtype=np.int64):
    """
    Similar to np.arange or torch.arange
    """
    if end is None:
        end = start
        start = 0
    return Tensor(np.arange(start, end, step), requires_grad=requires_grad)


def brodcast_tensors(tensor1, tensor2):
    """
    Brodcasts two tensors to the same shape
    """
    try:
        data1 = tensor1.data
        data2 = tensor2.data
        expanded_t1, expanded_t2 = np.broadcast_arrays(data1, data2)

        expanded_x = Tensor(expanded_t1, requires_grad=tensor1.requires_grad)
        expanded_y = Tensor(expanded_t2, requires_grad=tensor2.requires_grad)

    except ValueError as e:
        print(f"Error: {e}")
        return None, None
    return expanded_x, expanded_y


def randn(shape: tuple, requires_grad=False):
    """
    Returns a tensor with normally distributed values
    """
    return Tensor(np.random.randn(*shape), requires_grad=requires_grad)

def where(condition, x=None, y=None):
    if x is None and y is None:
        return np.where(condition.data)
    else:
        x_data = x.data if isinstance(x, Tensor) else x
        y_data = y.data if isinstance(y, Tensor) else y
        condition_data = condition.data if isinstance(condition, Tensor) else condition
        return Tensor(np.where(condition_data, x_data, y_data))