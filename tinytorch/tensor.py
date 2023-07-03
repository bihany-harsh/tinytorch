import numpy as np

class Tensor:
    def __init__(self, data, dtype=np.float64):
        if isinstance(data, list):
            self.data = np.array(data, dtype=dtype)
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, (int , float)):
            self.data = np.array(data)
        else:
            raise TypeError("Invalid data type for Tensor")
        self.shape = list(self.data.shape)

    def __repr__(self):
        return f"Tensor({self.data})"
    
    def __add__(self, other):
        if isinstance(other, list):
            raise TypeError("Invalid data type for Tensor")
        elif isinstance(other, np.ndarray):
            return Tensor(self.data + other)
        elif isinstance(other, (int, float)):
            return Tensor(self.data + other)
        elif isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        if isinstance(other, list):
            raise TypeError("Invalid data type for Tensor")
        elif isinstance(other, np.ndarray):
            return Tensor(self.data * other)
        elif isinstance(other, (int, float)):
            return Tensor(self.data * other)
        elif isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __pow__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self.data ** other)
        elif isinstance(other, Tensor):
            if self.shape != other.shape:
                raise TypeError("Invalid shape for Tensor")
            return Tensor(self.data ** other.data)
        else:
            raise TypeError("Invalid data type for Tensor")
        
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def exp(self):
        return Tensor(np.exp(self.data))


def arange(start=0, end=None, step=1, dtype=np.float64):
    if end is None:
        end = start
        start = 0
    return Tensor(np.arange(start, end, step))