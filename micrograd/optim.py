
from . import DTYPE

from .engine import Value
from numpy import zeros, sqrt


class SGD:
    ''' Sochastic gradient descent optimiser.

    velocity = momentum * velocity + g
    w = w - learning_rate * velocity

    https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    https://github.com/keras-team/keras/blob/master/keras/src/optimizers/sgd.py
    '''

    def __init__(self, wrt=[], learning_rate=None, momentum=None):
        '''
        wrt: a list of Values, with respect to which to minimise
            the target quantity.
        learning_rate: a non-negative number or a generator of them.
        momentum: None or a non-negative number.
        '''
        assert all([isinstance(_, Value) for _ in wrt])
        assert momentum is None or momentum >= 0

        self.wrt = wrt
        self.learning_rate = learning_rate
        self.momentum = momentum
        if momentum:
            self.velocity = [zeros(_.shape, dtype=DTYPE)
                             for _ in wrt]

    def step(self):
        ''' One step of stochastic gradient descent '''
        lr = (next(self.learning_rate) if hasattr(self.learning_rate,
                                                  '__next__')
              else self.learning_rate)
        assert lr >= 0

        if self.momentum is None:
            for v in self.wrt:
                v.data -= lr * v.grad
        else:
            for j, v in enumerate(self.wrt):
                self.velocity[j] *= self.momentum
                self.velocity[j] += v.grad
                v.data -= lr * self.velocity[j]


class ADAM:
    def __init__(self, wrt=[], learning_rate=None,
                 beta1=None, beta2=None, eps_adam=None):
        ''' eg learning_rate = .001, beta1 = .9,
                beta2 = .999, eps_adam = 1e-8,

            or learning_rate = .01, beta1 = .85,
                beta2 = .99, eps_adam = 1e-8
        '''
        assert beta1 >= 0
        assert beta2 >= 0
        assert eps_adam >= 0

        self.wrt = wrt
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps_adam = eps_adam
        # first moments
        self.m = [zeros(_.shape, dtype=DTYPE) for _ in wrt]
        # second moments
        self.v = [zeros(_.shape, dtype=DTYPE) for _ in wrt]
        self.num_steps = 0

    def step(self):
        lr = (next(self.learning_rate) if hasattr(self.learning_rate,
                                                  '__next__')
              else self.learning_rate)
        assert lr >= 0

        self.num_steps += 1
        for j, v in enumerate(self.wrt):
            self.m[j] *= self.beta1
            self.m[j] += (1 - self.beta1) * v.grad
            self.v[j] *= self.beta2
            self.v[j] += (1 - self.beta2) * v.grad ** 2
            m_hat = self.m[j] / (1 - self.beta1 ** self.num_steps)
            v_hat = self.v[j] / (1 - self.beta2 ** self.num_steps)
            v.data -= lr * m_hat / (sqrt(v_hat) + self.eps_adam)
