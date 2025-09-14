
from . import DTYPE

from .engine import Value
from numbers import Number
from numpy import zeros


class SGD:
    ''' Sochastic gradient descent optimiser.

    velocity = momentum * velocity + g
    w = w - learning_rate * velocity

    https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    https://github.com/keras-team/keras/blob/master/keras/src/optimizers/sgd.py
    '''

    def __init__(self, target, wrt=[], learning_rate=None, momentum=None):
        '''
        target: Value, the target variable to minimise. When the
            target variable is a tensor, the quantity to minimise is
            as if rewritten as the sum of each element of it.
        wrt: a list of Values, with respect to which to minimise
            the target quantity.
        learning_rate: a non-negative number or a generator of them.
        momentum: None or a non-negative number.
        '''
        assert isinstance(target, Value)
        assert isinstance(wrt, (list, tuple))
        assert all([isinstance(_, Value) for _ in wrt])
        if isinstance(learning_rate, Number):
            assert learning_rate >= 0
        else:
            assert hasattr(learning_rate, '__next__')
        assert momentum is None or momentum >= 0

        self.target = target
        self.wrt = wrt
        self.learning_rate = learning_rate
        self.learning_rate_is_number = isinstance(learning_rate, Number)
        self.momentum = momentum
        if momentum is not None:
            self.velocity = [zeros(_.shape, dtype=DTYPE)
                             for _ in wrt]

    def step(self, **kwds):
        ''' One step of stochastic gradient descent '''
        self.target.forward(**kwds)
        self.target.backward()

        if self.learning_rate_is_number:
            lr = self.learning_rate
        else:
            lr = next(self.learning_rate)
            assert lr >= 0

        if self.momentum is None:
            for v in self.wrt:
                v.data -= lr * v.grad
        else:
            for j, v in enumerate(self.wrt):
                self.velocity[j] *= self.momentum
                self.velocity[j] += v.grad
                v.data -= lr * self.velocity[j]
