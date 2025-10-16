# Tinytorch

Tinytorch is a lightweight implementation of PyTorch, a popular deep learning framework. It provides a simplified version of key functionality found in PyTorch, allowing users to understand the underlying concepts and build basic neural networks. It borrows ideas and implementations from Andrej Karpathy's youtube lecture on Micrograd.

## Features

- Tensor operations: Perform mathematical operations on multidimensional arrays. ✅
- Automatic differentiation: Track gradients and perform backpropagation. ✅
- Optimizer classes for carrying gradient descent and other optimization techniques. ✅
- Basic loss calculation, NLL loss, MSE loss, CrossEntropy loss and such. ✅
- Basic neural network modules: Implement layers such as Linear, activations like ReLU present. ✅
- Support for Modules and Parameters ✅
- Saving `state_dict()` functionality for Modules and Optimizers. ✅

## What's next:

- Support to print computation graph
- Temporal architecture
- Schedulers
- Data management in batches, data module setup.
- Sequential API

## Getting Started

### Prerequisites

- Python 3.6 or above
- Numpy 1.24 or above
- dill (for pickling the objects)

### Installation

- To setup `tinytorch` clone the repo, navigate to the directory containing `setup.py` file and then run the following command: `<br/>`

```sh
pip install -e .
```

Before installing, make sure your environment meets the prerequisites mentioned above. If you encounter any issues during installation, feel free to raise an issue on our GitHub repository.

Enjoy using Tinytorch!
