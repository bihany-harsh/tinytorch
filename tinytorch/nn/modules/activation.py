from tinytorch.nn import functional as F

class ReLU:
    def __init__(self):
        pass

    def __call__(self, x):
        self.x = x
        return F.relu(x)
    
class Tanh:
    def __init__(self):
        pass

    def __call__(self, x):
        self.x = x
        return F.tanh(x)
    
class Sigmoid:
    def __init__(self):
        pass

    def __call__(self, x):
        self.x = x
        return F.sigmoid(x)

class Softmax:
    def __init__(self):
        pass

    def __call__(self, x):
        self.x = x
        return F.softmax(x)
    
class LogSoftmax:
    def __init__(self):
        pass

    def __call__(self, x):
        self.x = x
        return F.log_softmax(x)