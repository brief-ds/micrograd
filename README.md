
# micrograd
A tiny Autograd engine whose only dependency is NumPy. Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny.

* 24 kilobytes code without tests
* as portable as Python and NumPy
* loads 10x faster than PyTorch, 30x faster than TensorFlow

This version is capable of working with matrices and higher-order tensors. For @karpathy's original scalar-based version, locate the code with tag `scalar`.

## Installation
### For deployment
Inside the external project where you want to deploy and use micrograd,

```bash
python3 -m venv venv
. venv/bin/activate
cd <path-to-micrograd-project>
pip3 install .
```

### For development
As different test files have different requirements, one may set up a virtual environment `venv` just for `tests/test_engine.py`, and a separate environment `torch` for all `tests/*.py` including `tests/test_vs_torch.py`, which requires install of PyTorch. No need to run `pip3 install .` under either environment (for running the tests). For example,

```bash
python3 -m venv torch
. torch/bin/activate

# no need to run "pip3 install ."
# but PyTorch need be installed
# into the torch virtual environment
pip3 install torch
```

### For running the demos under `demos/`
Create a third virtual environment `jupyter`. Install the requirements and the micrograd package itself.

```bash
python3 -m venv jupyter
. jupyter/bin/activate

pip3 install .
pip3 install jupyter
```

## Get Started
```python
from micrograd import Value
from numpy import array

a = Value(array([[2, 3], [5, 4]]))
b = Value(array([1, -1]))
c = (a @ b).relu()
print(c)      # Value(data=[0 1], grad=None)
c.backward()
print(c)      # Value(data=[0 1], grad=[1. 1.])
print(a)      # Value(data=..., grad=[[0. 0.], [1. -1.]])
print(b)      # Value(data=..., grad=[5. 4.])
```

PyTorch requires any expression to be derived with respect to variables to yield a scalar. micrograd relaxes it: it starts with an all-ones tensor of the shape of the expression's result, as if rewriting the quantity to be derived as the sum of each element of the expression's original result.

## Data type
As one example, with `f=ab`, `df/da=b`. `a.grad` would inherit the data type of `b`. For this inter-dependence, we design a uniform `DTYPE` for one program using micrograd, to be passed from the environment. By default `DTYPE=float64`, identical as the Python float type. For example,

```sh
DTYPE=float32 python3 <program_using_micrograd>
```

micrograd's `__init__.py` reads `DTYPE` from the environment. In Python, _before_ importing micrograd, one may manipulate the `DTYPE` by

```python
from os import environ
environ['DTYPE'] = ...

from micrograd import Value
```

One may get the `DTYPE` that micrograd read,

```python
from micrograd import DTYPE
```

## Lazy evaluation
When defining a tensor, one may just indicate `shape` and `name`, and later on provide the value.

```python
from micrograd import Value
from numpy import array

a = Value(shape=(2, 2), name='var1')
b = Value(shape=(2,), name='var2')
c = (a @ b).relu()
c.forward(var1=array([[2, 3], [5, 4]]),
          var2=array([1, -1]))
c.backward()
```

## Essential Use Pattern
Call `forward()` once with the values for the varialbes, then `backward()` once for the mathematical derivatives.

```python
x.forward(var1=value1, var2=value2, ...)
x.backward()
```

Each time the `forward()` is called (e.g. for minibatch evaluation), the lazily defined variables have to be fed values in the function signature. Otherwise, it will take all `nan` as value. The final result will likely be `nan` to signal missing values for some variables.

If an expression has no lazy variables at all, `forward()` call is not necessary. Once defined, the expression is evaluated.

Inside the `backward()` call, all the derivatives are initialised zero other than the final one to be initialised all-one, before the chain derivation. So no `zero_grad()` is necessary or defined anywhere.

## Efficient operator dependency topology computation
The operator dependency topology is only calculated once then cached, supposing the topology is *static* once an expression is defined.

## Supported operators
* `__pow__`
* `__matmul__`
* `tensordot` for tensor contraction: unlike numpy tensordot, the last axis (indexed by -1) of the left tensor contracts with the first axis of the right tensor; the next to last axis (indexed by -2) of the left tensor with the 2nd axis of the right tensor; so on and so forth.
* `relu`
* `log`
* `log1p`
* `tanh`
* `arctanh`
* `T` for transpose
* `sum`
* `mean`

## Stochastic Gradient Descent
To be able to implement any SGD algorithm flexibly, the `micrograd.optim.SGD` is designed as

```python
SGD(target,   # variable to be minimised
    wrt=[],   # list of variables with respect to which to perform minimisation
    learning_rate=None,    # a non-negative number or a generator of them
    <SGD param>,
    <SGD param>, ...)
```

When the target variable is not a scalar, the objective function is as if rewritten as the sum of each element in the target tensor. The `learning_rate` can accept a generator implementing a schedule of varying learning rates.

Once initialised, just call `step()` on the optimiser with the minibatch data.

```python
optimiser = SGD(...)

# batch_iterator yields a dict
# for the minibatch, e.g.
#
#   {'X': .., 'y': ..}

for k in range(..):
    # one step of gradient descent on all parameters
    batch_data = next(batch_iterator)
    optimiser.step(**batch_data)

    # one may now call target.forward(..) for validation
```

Refer to `micrograd/optim.py` for more detail.

## The Demos
The notebooks under `demos/` provide a full demo of training an 2-layer neural network (MLP) binary classifier. This is achieved by initializing a neural net from `micrograd.nn` module, implementing a simple svm "max-margin" binary classification loss and using SGD for optimization. As shown in the notebook, using a 2-layer neural net with two 16-node hidden layers we achieve the following decision boundary on the moon dataset:

![2d neuron](assets/moon_mlp.png)

## Tracing / visualization
For added convenience, the notebook `trace_graph.ipynb` produces graphviz visualizations. E.g. this one below is of a simple 2D neuron, arrived at by calling `draw_dot` on the code below, and it shows both the data (left number in each node) and the gradient (right number in each node).

```python
from micrograd import nn
n = nn.Neuron(2)
x = [Value(1.0), Value(-2.0)]
y = n(x)
dot = draw_dot(y)
```

![2d neuron](assets/gout.svg)

## Running tests
The following line uses the built-in `unittest` module to run the unit tests. But to run `tests/test_vs_torch.py` it requires PyTorch. One could create a separate virtual environment for test, as PyTorch may require downgrade of NumPy to version 1.

```bash
python -m unittest tests/*.py
```

## License
MIT
