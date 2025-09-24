
# micrograd
A tiny autograd engine whose only dependency is NumPy the linear algebra library. Micrograd implements backpropagation (automatic differentiation) over a graph of mathemtical operations.

* 20 kilobytes core code, 10,000+ times smaller
* as portable as Python and NumPy
* comparable performance as mammoth contenders
* code can be timed with Python's native profiler

This version works with vectors, including matrices (2-dimensional), or higher-dimensional tensors. For @karpathy's original scalar-based version, switch to the code with tag `scalar`.

## Get Started
In any working directory, create a virtual environment,

```sh
python3 -m venv venv
. venv/bin/activate
cd <directory_of_micrograd>     # if not already in the micrograd's directory
pip3 install .
cd <initial_working_directory>  # if different from micrograd
pip3 install jupyter            # for running demos in demos/
pip3 install torch              # to run tests/test_vs_torch.py
```

Below is a Python snippet using micrograd,

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

PyTorch can only mathematically derive an expression that produces a scalar value. micrograd relaxes it: if the expression produces an array, the sum of the array will be derived.

For full examples, go to [`demos/`](demos). The scalar-version [demos/demo_scalar.ipynb](demos/demo_scalar.ipynb) takes minutes to run, but the vector-version training [demos/demo_vector.ipynb](demos/demo_vector.ipynb) is instant.

## Data type
As one example, with `f=ab`, `df/da=b`. `a.grad` would inherit the data type of `b`. For this inter-dependence, we design a uniform `DTYPE` for one program, to be passed from the environment. By default `DTYPE=float64`, identical as the Python float type. For example,

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
