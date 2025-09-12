from tinytorch import tensor
from collections import OrderedDict

class Module:
    def __init__(self):
        
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "_training", True)
    
    # CORE API
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the forward pass")
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    # REGISTRATION
    
    def register_parameter(self, name, param):
        if param is not None:
            self._parameters[name] = param

    def register_buffer(self, name, tensor_):
        if tensor_ is not None and not isinstance(tensor_, tensor.Tensor):
            raise TypeError("buffer should be a Tensor")
        self._buffers[name] = tensor_
        
    # ITERATORS
    
    def parameters(self):
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def named_parameters(self, prefix="", recurse=True):
        for n, p in self._parameters.items():
            yield prefix + n, p
        if recurse:
            for n, m in self._modules.items():
                yield from m.named_parameters(prefix + n + ".", recurse=True)

    def buffers(self):
        for b in self._buffers.values():
            yield b
        for m in self._modules.values():
            yield from m.buffers()

    def named_buffers(self, prefix="", recurse=True):
        for n, b in self._buffers.items():
            yield prefix + n, b
        if recurse:
            for n, m in self._modules.items():
                yield from m.named_buffers(prefix + n + ".", recurse=True)
            
    def children(self):
        return self._modules.values()
    
    def named_children(self):
        return self._modules.items()
            
    # STATE DICT I/O
    
    def state_dict(self):
        return {name: p for name, p in self.named_parameters()}
    
    def load_state_dict(self, sd):
        missing, unexpected = [], []

        # parameters
        for name, tgt in self.named_parameters():
            if name in sd:
                self._assign(tgt, sd[name])
            else:
                missing.append(name)

        # buffers
        for name, tgt in self.named_buffers():
            if name in sd:
                self._assign(tgt, sd[name])
            else:
                missing.append(name)

        # extras
        unexpected = [k for k in sd.keys()
                      if k not in dict(self.named_parameters())
                      and k not in dict(self.named_buffers())]

        if missing or unexpected:
            print("load_state_dict warnings:",
                  "missing", missing, "unexpected", unexpected)
    
    @staticmethod
    def _assign(dst, src):
        if not isinstance(dst, tensor.Tensor) or not isinstance(src, tensor.Tensor):
            raise TypeError("Both dst and src must be Tensor objects")

        if dst.data.shape != src.data.shape:
            raise ValueError(f"Shape mismatch in state_dict load: "
                            f"dst.shape={dst.data.shape}, src.shape={src.data.shape}")

        dst.data[...] = src.data.astype(dst.data.dtype, copy=False)
        
    # TRAIN/EVAL
    
    def train(self, mode=True):
        self._training = mode
        for m in self._modules.values():
            m.train(mode)
        return self
    
    def eval(self):
        return self.train(False)
    
    # ATTR ACCESS
    def __setattr__(self, name, value):
        from tinytorch.tensor import Tensor
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Tensor):
            self._parameters[name] = value
        
        object.__setattr__(self, name, value)
        
    def __getattr__(self, name):
        if name in self._parameters:
            return self._parameters[name]
        if name in self._buffers:
            return self._buffers[name]
        if name in self._modules:
            return self._modules[name]
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")
    
    # tinytorch/nn/module.py
    def __repr__(self) -> str:
        main_str = self._get_name() + '('
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _add_indent(mod_str, 2)         # indent child block
            child_lines.append(f"({key}): {mod_str}")
        if child_lines:
            main_str += '\n  ' + '\n  '.join(child_lines) + '\n'
        main_str += ')'
        return main_str

    # helper
    def _get_name(self):
        return self.__class__.__name__

def _add_indent(s, num_spaces):
    indented = []
    for line in s.split('\n'):
        if line.strip():
            line = ' ' * num_spaces + line
        indented.append(line)
    return '\n'.join(indented)

        
        
        
        