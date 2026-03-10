
from . import DTYPE

from numpy import (array, ndarray, nan,
                   ones, zeros, full,
                   shape as np_shape, where,
                   max as np_max,
                   maximum, take, prod,
                   exp, log, log1p, tanh,
                   arctanh, arcsin,
                   transpose, sum as np_sum,
                   tensordot as np_tensordot,
                   broadcast_to, expand_dims,
                   concatenate as np_concatenate, split,
                   isnan, all as np_all)
from numbers import Number
from warnings import warn

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data=None, _children=(), _op='',
                 shape=None, name=None):
        if data is not None:
            assert isinstance(data, (ndarray, Number))
            assert name is None
            assert shape is None
            self.name = None
            self.shape = np_shape(data)
            # dtype must be enforced on non-scalar data
            self.data = data.astype(DTYPE) if self.shape else data
        else:
            assert name
            assert isinstance(shape, tuple)
            self.name = name
            self.shape = shape
            self.data = full(shape, nan, dtype=DTYPE)
        self.grad = None
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

        def _forward(**kwds):
            if self.name:
                if self.name in kwds:
                    _value = kwds[self.name]
                    assert isinstance(_value, (ndarray, Number))
                    assert np_shape(_value) == self.shape
                    # dtype must be enforced on non-scalar
                    self.data = (_value.astype(DTYPE) if self.shape
                                 else _value)
                else:
                    warn(f'{self.name} not in input data')
                    self.data = full(self.shape, nan, dtype=DTYPE)
        self._forward = _forward

    def __add__(self, other):
        other = (other if isinstance(other, Value)
                 else Value(other, _op='c'))
        out = Value(self.data + other.data, (self, other), '+')

        def _forward(**kwds):
            out.data = self.data + other.data
        out._forward = _forward

        def _backward():
            # in some cases, the shape of one operand
            # would have been broadcast to higher dimensions
            if self.ndim < out.ndim:
                self.grad += (out.grad
                              .sum(axis=tuple(range(out.ndim - self.ndim))))
            else:
                self.grad += out.grad
            if other.ndim < out.ndim:
                other.grad += (out.grad
                               .sum(axis=tuple(range(out.ndim - other.ndim))))
            else:
                other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = (other if isinstance(other, Value)
                 else Value(other, _op='c'))
        out = Value(self.data * other.data, (self, other), '*')

        def _forward(**kwds):
            out.data = self.data * other.data
        out._forward = _forward

        def _backward():
            # in some cases, the shape of one operand
            # would have been broadcast to higher dimensions
            if self.ndim < out.ndim:
                self.grad += ((other.data * out.grad)
                              .sum(axis=tuple(range(out.ndim - self.ndim))))
            else:
                self.grad += other.data * out.grad
            if other.ndim < out.ndim:
                other.grad += ((self.data * out.grad)
                               .sum(axis=tuple(range(out.ndim - other.ndim))))
            else:
                other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        # TOODO: array(3) ** -1 won't do. array(3).astype(float) if excepted
        assert isinstance(other, (int, float)), ("only supporting"
                                                 " int/float powers for now")
        out = Value(self.data ** other, (self,), f'**{other}')

        def _forward(**kwds):
            out.data = self.data ** other
        out._forward = _forward

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    @property
    def T(self):
        out = Value(transpose(self.data), (self,), 'T')

        def _forward(**kwds):
            out.data = transpose(self.data)
        out._forward = _forward

        def _backward():
            self.grad += transpose(out.grad)
        out._backward = _backward

        return out

    @property
    def ndim(self):
        return len(self.shape)

    def topk(self, k):
        assert self.ndim == 1
        k2 = self.data.size - k

        pargs = argpartition(self.data, k2)
        out = Args(pargs[-k:], (self,), 'topk')

        def _forward(**kwds):
            pargs = argpartition(self.data, k2)
            out.data = pargs[-k:]
        out._forward = _forward

        return out

    def attend(self, args):
        if isinstance(args, Args):
            out = Value(self.data[args.data], (self, args), 'attend')
        else:
            out = Value(self.data[args], (self,), 'attend')

        def _forward(**kwds):
            _args = args.data if isinstance(args, Args) else args
            out.data = self.data[_args]
        out._forward = _forward

        def _backward():
            _args = args.data if isinstance(args, Args) else args
            self.grad[_args] += out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(maximum(self.data, 0), (self,), 'ReLU')

        def _forward(**kwds):
            out.data = maximum(self.data, 0)
        out._forward = _forward

        def _backward():
            self.grad += where(out.data > 0, out.grad, 0)
        out._backward = _backward

        return out

    def exp(self):
        out = Value(exp(self.data), (self,), 'exp')

        def _forward(**kwds):
            out.data = exp(self.data)
        out._forward = _forward

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def log(self):
        out = Value(log(self.data), (self,), 'log')

        def _forward(**kwds):
            out.data = log(self.data)
        out._forward = _forward

        def _backward():
            valid_data = where(self.data >= 0, self.data, nan)
            self.grad += 1 / valid_data * out.grad
        out._backward = _backward

        return out

    def log1p(self):
        out = Value(log1p(self.data), (self,), 'log1p')

        def _forward(**kwds):
            out.data = log1p(self.data)
        out._forward = _forward

        def _backward():
            valid_data = where(self.data >= -1, self.data, nan)
            self.grad += 1 / (1 + valid_data) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        out = Value(tanh(self.data), (self,), 'tanh')

        def _forward(**kwds):
            out.data = tanh(self.data)
        out._forward = _forward

        def _backward():
            self.grad += (1 - tanh(self.data) ** 2) * out.grad
        out._backward = _backward

        return out

    def arctanh(self):
        out = Value(arctanh(self.data), (self,), 'arctanh')

        def _forward(**kwds):
            out.data = arctanh(self.data)
        out._forward = _forward

        def _backward():
            valid_data = where((-1 <= self.data) & (self.data <= 1),
                               self.data, nan)
            self.grad += 1 / (1 - valid_data ** 2) * out.grad
        out._backward = _backward

        return out

    def arcsin(self):
        out = Value(arcsin(self.data), (self,), 'arcsin')

        def _forward(**kwds):
            out.data = arcsin(self.data)
        out._forward = _forward

        def _backward():
            valid_data = where((-1 <= self.data) & (self.data <= 1),
                               self.data, nan)
            self.grad += 1 / (1 - valid_data ** 2) ** .5 * out.grad
        out._backward = _backward

        return out

    def sum(self, axis=None):
        # map any negative dimension index to non-negative one
        de_neg = lambda x: self.ndim + x if x < 0 else x

        if axis is None:
            _axis = tuple(range(self.ndim))
        elif isinstance(axis, int):
            _axis = de_neg(axis)
        else:
            _axis = tuple(map(de_neg, axis))

        out = Value(np_sum(self.data, axis=axis), (self,), 'sum')

        def _forward(**kwds):
            out.data = np_sum(self.data, axis=axis)
        out._forward = _forward

        def _backward():
            # expand out.grad to same number of dimensions
            # as self.data, self.grad
            _out_grad = expand_dims(out.grad, _axis)

            # ... expand further to same shape as self.data
            self.grad += broadcast_to(_out_grad, self.shape)
        out._backward = _backward

        return out

    def mean(self, axis=None):
        if axis is None:
            denom = prod(self.shape)
        else:
            denom = prod(take(self.shape, axis))
        return self.sum(axis) * (1 / denom)

    def max(self, axis=None):
        if axis is not None:
            raise NotImplementedError

        out = Value(np_max(self.data, axis=axis), (self,), 'max')

        def _forward(**kwargs):
            out.data = np_max(self.data, axis=axis)
        out._forward = _forward

        def _backward():
            exp_data = out.data
            self.grad[self.data == exp_data] += out.grad
        out._backward = _backward

        return out

    def softmax(self, axis=None):
        if axis is not None:
            raise NotImplementedError

        self_max = self.max(axis=axis)
        y = self - self_max
        y_exp = y.exp()
        y_exp_sum = y_exp.sum(axis=axis)
        return y_exp / y_exp_sum

    def build_topology(self):
        # topological order all of the children in the graph
        if not hasattr(self, 'topo'):
            self.topo = []
            visited = set()

            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build_topo(child)
                    self.topo.append(v)

            build_topo(self)

    def forward(self, **kwds):

        self.build_topology()

        for v in self.topo:
            v._forward(**kwds)

    def backward(self):

        if np_all(isnan(self.data)):
            warn('run forward() before backward()')

        self.build_topology()

        # go one variable at a time and
        # apply the chain rule to get its gradient
        for v in self.topo:
            if not isinstance(v, Value):
                continue
            if v.grad is None:       # array not allocated yet
                v.grad = (ones(v.shape, dtype=DTYPE) if v == self
                          else zeros(v.shape, dtype=DTYPE))
            else:                    # array has been allocated
                v.grad.fill(1 if v == self else 0)
        for v in reversed(self.topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __matmul__(self, other):
        return tensordot(self, other, 1)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


class Args:

    def __init__(self, data=None, _children=(), _op=''):
        assert isinstance(data, (ndarray, Number))
        self.name = None
        self.shape = _shape(data)
        # dtype must be enforced on non-scalar data
        self.data = data.astype(int) if self.shape else int(data)
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._forward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __repr__(self):
        return f"Value(data={self.data})"


def tensordot(left, right, axes):
    ''' Tensor contraction, only accepting int axes

    Example use:

        tensordot(left, right, axes=2)

    Unlike numpy tensordot, the last axis (indexed by -1) of the left
    tensor contracts with the first axis of the right tensor; the
    next to last axis (indexed by -2) of the left tensor with the 2nd
    axis of the right tensor; so on and so forth.
    '''
    assert axes >= 0          # only int axes
    assert axes <= left.ndim
    assert axes <= right.ndim

    # axes for various numpy tensordot ops later
    axes1 = ([-1 - j for j in range(axes)], list(range(axes)))
    axes2 = ([-1 - j for j in range(left.ndim - axes)],
             list(range(left.ndim - axes)))
    axes3 = ([-1 - j for j in range(right.ndim - axes)],
             list(range(right.ndim - axes)))

    left = (left if isinstance(left, Value)
            else Value(left, _op='c'))
    right = (right if isinstance(right, Value)
             else Value(right, _op='c'))
    out = Value(np_tensordot(left.data, right.data, axes=axes1),
                (left, right), '@')

    def _forward(**kwds):
        out.data = np_tensordot(left.data, right.data, axes=axes1)
    out._forward = _forward

    def _backward():
        left.grad += np_tensordot(out.grad, transpose(right.data),
                                axes=axes3)
        right.grad += np_tensordot(transpose(left.data), out.grad,
                                 axes=axes2)
    out._backward = _backward

    return out

def concatenate(lst, axis):
    lst = [_ if isinstance(_, Value) else Value(_)
           for _ in lst]
    widths = [_.shape[axis] for _ in lst]

    out = Value(np_concatenate([_.data for _ in lst], axis=axis),
                tuple(lst), 'concat')

    def _forward(**kwds):
        out.data = np_concatenate([_.data for _ in lst], axis=axis)
    out._forward = _forward

    def _backward():
        lst_grad = split(out.grad, widths, axis=axis)
        for v, grad in zip(lst, lst_grad):
            v.grad += grad
    out._backward = _backward

    return out
