
from numpy import (array, ndarray, nan, ones, zeros, full,
                   shape as _shape, where,
                   log, log1p, tanh, arctanh,
                   broadcast_arrays, expand_dims,
                   prod, tensordot as _tensordot,
                   isnan, all as npall)
from numbers import Number
from warnings import warn

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data=None, _children=(), _op='',
                 shape=None, name=None):
        if data is not None:
            assert isinstance(data, (ndarray, Number))
            assert name is None, "data provided, no need for name"
            assert shape is None, "data provided, no need for shape"
            self.data = data
            self.name = None
            self.shape = _shape(data)
        else:
            assert name, "data not provided, name must be given"
            assert isinstance(shape, tuple), "shape must be given"
            self.name = name
            self.shape = shape
            self.data = full(self.shape, nan)
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
                    assert _shape(_value) == self.shape
                    self.data = _value
                else:
                    warn(f'{self.name} not in input data')
                    self.data = full(self.shape, nan)
        self._forward = _forward

    def __add__(self, other):
        other = (other if isinstance(other, Value)
                 else Value(other, _op='c'))
        out = Value(self.data + other.data, (self, other), '+')

        def _forward(**kwds):
            out.data = self.data + other.data
        out._forward = _forward

        def _backward():
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
        out = Value(self.data.T, (self,), 'T')

        def _forward(**kwds):
            out.data = self.data.T
        out._forward = _forward

        def _backward():
            self.grad += out.grad.T
        out._backward = _backward

        return out

    @property
    def ndim(self):
        return len(self.shape)

    def relu(self):
        out = Value(where(self.data > 0, self.data, 0), (self,), 'ReLU')

        def _forward(**kwds):
            out.data = where(self.data > 0, self.data, 0)
        out._forward = _forward

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def log(self):
        out = Value(log(self.data), (self,), 'log')

        def _forward(**kwds):
            out.data = log(self.data)
        out._forward = _forward

        def _backward(**kwds):
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
            arctanh_grad = 1 / (1 - self.data ** 2)
            arctanh_grad = where(arctanh_grad >= 1, arctanh_grad, nan)
            self.grad += arctanh_grad * out.grad
        out._backward = _backward

        return out

    def sum(self, axis=None):
        if self.ndim == 0:
            assert not axis
            return self

        out = Value(self.data.sum(axis=axis), (self,), 'sum')

        de_neg = lambda x: self.ndim + x if x < 0 else x
        if axis is None:
            expand_axis = tuple(range(self.data.ndim))
        elif isinstance(axis, int):
            expand_axis = de_neg(axis)
        else:
            expand_axis = tuple(map(de_neg, axis))

        arr_orig_shape = ones(self.shape)

        def _forward(**kwds):
            out.data = self.data.sum(axis=axis)
        out._forward = _forward

        def _backward():
            self.grad += broadcast_arrays(expand_dims(out.grad, expand_axis),
                                          arr_orig_shape)[0]
        out._backward = _backward

        return out

    def mean(self, axis=None):
        shape_arr = array(self.shape)
        if axis is None:
            denom = prod(shape_arr)
        elif isinstance(axis, int):
            denom = shape_arr[axis]
        else:
            denom = prod(shape_arr[list(axis)])

        return self.sum(axis) * (1 / denom)

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

        if npall(isnan(self.data)):
            warn('run forward() before backward()')

        self.build_topology()
        # go one variable at a time and apply the chain rule to get its gradient
        for v in self.topo:
            v.grad = ones(self.shape) if v == self else zeros(v.shape)
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
    axes1 = ([-1 - j for j in range(axes)], list(range(axes)))
    axes2 = ([-1 - j for j in range(left.ndim - axes)],
             list(range(left.ndim - axes)))
    axes3 = ([-1 - j for j in range(right.ndim - axes)],
             list(range(right.ndim - axes)))

    left = (left if isinstance(left, Value)
            else Value(left, _op='c'))
    right = (right if isinstance(right, Value)
             else Value(right, _op='c'))
    out = Value(_tensordot(left.data, right.data, axes=axes1),
                (left, right), '@')

    def _forward(**kwds):
        out.data = _tensordot(left.data, right.data, axes=axes1)
    out._forward = _forward

    def _backward():
        left.grad += _tensordot(out.grad, right.data.T, axes=axes3)
        right.grad += _tensordot(left.data.T, out.grad, axes=axes2)
    out._backward = _backward

    return out
