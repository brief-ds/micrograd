from unittest import TestCase
from micrograd import Value, tensordot
from numpy import array, isclose, allclose, nan, inf, empty, pi

class AutodiffTest(TestCase):

    def test_sanity_check(self):

        a = Value(shape=(2,), name='a')
        a.forward(a=array([2, 3]))
        a.backward()
        self.assertTrue(allclose(a.grad, [1, 1]))

    def test_unary_ops(self):
        # test arithmetic, relu
        a = Value(shape=(2,), name='a')
        b = Value(shape=(2,), name='b')
        c = (a + 2).relu() * b ** 2
        c.forward(a=array([-3, 1]), b=array([2, 3]))
        c.backward()
        self.assertTrue(allclose(c.data, [0, 27]))
        self.assertTrue(allclose(a.grad, [0, 9]))
        self.assertTrue(allclose(b.grad, [0, 18]))
        self.assertTrue(allclose(c.grad, [1, 1]))

        # test log
        d = a.log()
        d.forward(a=array([2, 3]))
        d.backward()
        self.assertTrue(allclose(d.data, [0.69314718, 1.09861229]))
        self.assertTrue(allclose(a.grad, [.5, 1 / 3]))

        # test log1p
        d = a.log1p()
        d.forward(a=array([2, 3]))
        d.backward()
        self.assertTrue(allclose(d.data, [1.09861229, 1.38629436]))
        self.assertTrue(allclose(a.grad, [0.33333333, 0.25]))

        d.forward(a=array([-2, 3]))
        d.backward()
        self.assertTrue(allclose(d.data, [nan, 1.38629436], equal_nan=True))
        self.assertTrue(allclose(a.grad, [nan, 0.25], equal_nan=True))

        # test transpose
        f = Value(shape=(2, 1), name='f')
        g = f.T ** 2
        g.forward(f=array([[2], [-1]]))
        g.backward()
        self.assertTrue(allclose(f.grad, [[4], [-2]]))

        # test tanh
        k = a.tanh()
        k.forward(a=array([0, 2]))
        k.backward()
        self.assertTrue(allclose(k.data, [0., 0.96402758]))
        self.assertTrue(allclose(a.grad, [1., 0.07065082]))

        # test arctanh
        h = Value(shape=(5,), name='h')
        k = (h * 2).arctanh()
        k.forward(h=array([-1, -.5, 0, .5, 1]))
        k.backward()
        self.assertTrue(allclose(h.grad, [nan, inf, 2, inf, nan],
                                 equal_nan=True))

        # test arcsin
        k = h.arcsin()
        k.forward(h=array([-1.01, 0, 1 / 2 ** .5, 1, 1.01]))
        k.backward()
        self.assertTrue(allclose(k.data, [nan, 0, pi / 4, pi / 2, nan],
                                 equal_nan=True))
        self.assertTrue(allclose(h.grad, [nan, 1, 2 ** .5, inf, nan],
                                 equal_nan=True))

    def test_unary_ops_scalar_input(self):

        a = Value(shape=(), name='a')

        # test log
        d = a.log()
        d.forward(a=3)
        d.backward()
        self.assertTrue(allclose(d.data, 1.09861229))
        self.assertTrue(allclose(a.grad, 1 / 3))

        # test log1p
        d = a.log1p()
        d.forward(a=3)
        d.backward()
        self.assertTrue(allclose(d.data, 1.38629436))
        self.assertTrue(allclose(a.grad, 0.25))

        # test transpose
        g = a.T ** 2
        g.forward(a=3)
        g.backward()
        self.assertTrue(allclose(a.grad, 6))

        # test tanh
        k = a.tanh()
        k.forward(a=2)
        k.backward()
        self.assertTrue(allclose(k.data, 0.96402758))
        self.assertTrue(allclose(a.grad, 0.07065082))

        # test arctanh
        k = (a * 2).arctanh()
        k.forward(a=.5)
        k.backward()
        self.assertTrue(allclose(a.grad, inf))

        # test arcsin
        k = a.arcsin()
        k.forward(a=-1 / 2 ** .5)
        k.backward()
        self.assertTrue(allclose(a.grad, 2 ** .5))

    def test_sum_op(self):

        a = Value(shape=(2, 2, 3), name='a')
        b = (a.sum(axis=(0, 2)) - 31).relu()
        b.forward(a=array([[[1, 2, 3], [4, 5, 6]],
                           [[7, 8, 9], [10, 11, 12]]]))
        b.backward()
        self.assertTrue(allclose(b.data, [0, 17]))
        self.assertTrue(allclose(a.grad, [[[0, 0, 0], [1, 1, 1]],
                                          [[0, 0, 0], [1, 1, 1]]]))

        b = (a.sum(axis=0) - 10).relu()
        b.forward(a=array([[[1, 2, 3], [4, 5, 6]],
                           [[7, 8, 9], [10, 11, 12]]]))
        b.backward()
        self.assertTrue(allclose(a.grad, [[[0, 0, 1], [1, 1, 1]],
                                          [[0, 0, 1], [1, 1, 1]]]))

        b = (a.sum() - 77).relu()
        c = (a.sum() - 79).relu()
        b.forward(a=array([[[1, 2, 3], [4, 5, 6]],
                           [[7, 8, 9], [10, 11, 12]]]))
        b.backward()
        self.assertTrue(allclose(a.grad, 1))

        c.forward(a=array([[[1, 2, 3], [4, 5, 6]],
                           [[7, 8, 9], [10, 11, 12]]]))
        c.backward()
        self.assertTrue(allclose(a.grad, 0))

    def test_sum_op_neg_axis(self):

        a = Value(shape=(2, 2, 3), name='a')
        b = (a.sum(axis=(0, -1)) - 31).relu()
        b.forward(a=array([[[1, 2, 3], [4, 5, 6]],
                           [[7, 8, 9], [10, 11, 12]]]))
        b.backward()
        self.assertTrue(allclose(b.data, [0, 17]))
        self.assertTrue(allclose(a.grad, [[[0, 0, 0], [1, 1, 1]],
                                          [[0, 0, 0], [1, 1, 1]]]))

        b = (a.sum(axis=-3) - 10).relu()
        b.forward(a=array([[[1, 2, 3], [4, 5, 6]],
                           [[7, 8, 9], [10, 11, 12]]]))
        b.backward()
        self.assertTrue(allclose(a.grad, [[[0, 0, 1], [1, 1, 1]],
                                          [[0, 0, 1], [1, 1, 1]]]))

        b = (a.sum() - 77).relu()
        c = (a.sum() - 79).relu()
        b.forward(a=array([[[1, 2, 3], [4, 5, 6]],
                           [[7, 8, 9], [10, 11, 12]]]))
        b.backward()
        self.assertTrue(allclose(a.grad, 1))

        c.forward(a=array([[[1, 2, 3], [4, 5, 6]],
                           [[7, 8, 9], [10, 11, 12]]]))
        c.backward()
        self.assertTrue(allclose(a.grad, 0))

    def test_sum_op_scalar_input(self):

        a = Value(shape=(), name='a')
        b = (a ** 2).sum()
        b.forward(a=2.5)
        b.backward()
        self.assertTrue(allclose(b.data, 6.25))
        self.assertTrue(allclose(a.grad, 5))

        b = (a ** 2).sum(axis=())
        b.forward(a=2.5)
        b.backward()
        self.assertTrue(allclose(b.data, 6.25))
        self.assertTrue(allclose(a.grad, 5))

    def test_mean_op(self):

        a = Value(shape=(2, 2, 3), name='a')
        self.assertTrue(a.mean()._op == '*')

        b = (a.mean(axis=(0, 2)) - 6).relu()
        b.forward(a=array([[[1, 2, 3], [4, 5, 6]],
                           [[7, 8, 9], [10, 11, 12]]]))
        b.backward()
        y = 1 / 6
        self.assertTrue(allclose(b.data, [0, 2]))
        self.assertTrue(allclose(a.grad, [[[0, 0, 0], [y, y, y]],
                                          [[0, 0, 0], [y, y, y]]]))

        b = (a.mean(axis=0) - 5).relu()
        b.forward(a=array([[[1, 2, 3], [4, 5, 6]],
                           [[7, 8, 9], [10, 11, 12]]]))
        b.backward()
        y = .5
        self.assertTrue(allclose(a.grad, [[[0, 0, y], [y, y, y]],
                                          [[0, 0, y], [y, y, y]]]))

        b = (a.mean() - 6).relu()
        c = (a.mean() - 7).relu()
        b.forward(a=array([[[1, 2, 3], [4, 5, 6]],
                           [[7, 8, 9], [10, 11, 12]]]))
        b.backward()
        y = 1 / 12
        self.assertTrue(allclose(a.grad, y))

        c.forward(a=array([[[1, 2, 3], [4, 5, 6]],
                           [[7, 8, 9], [10, 11, 12]]]))
        c.backward()
        self.assertTrue(allclose(a.grad, 0))

    def test_mean_op_neg_axis(self):

        a = Value(shape=(2, 2, 3), name='a')
        self.assertTrue(a.mean()._op == '*')

        b = (a.mean(axis=(0, -1)) - 6).relu()
        b.forward(a=array([[[1, 2, 3], [4, 5, 6]],
                           [[7, 8, 9], [10, 11, 12]]]))
        b.backward()
        y = 1 / 6
        self.assertTrue(allclose(b.data, [0, 2]))
        self.assertTrue(allclose(a.grad, [[[0, 0, 0], [y, y, y]],
                                          [[0, 0, 0], [y, y, y]]]))

        b = (a.mean(axis=-3) - 5).relu()
        b.forward(a=array([[[1, 2, 3], [4, 5, 6]],
                           [[7, 8, 9], [10, 11, 12]]]))
        b.backward()
        y = .5
        self.assertTrue(allclose(a.grad, [[[0, 0, y], [y, y, y]],
                                          [[0, 0, y], [y, y, y]]]))

        b = (a.mean() - 6).relu()
        c = (a.mean() - 7).relu()
        b.forward(a=array([[[1, 2, 3], [4, 5, 6]],
                           [[7, 8, 9], [10, 11, 12]]]))
        b.backward()
        y = 1 / 12
        self.assertTrue(allclose(a.grad, y))

        c.forward(a=array([[[1, 2, 3], [4, 5, 6]],
                           [[7, 8, 9], [10, 11, 12]]]))
        c.backward()
        self.assertTrue(allclose(a.grad, 0))

    def test_mean_op_scalar_input(self):

        a = Value(shape=(), name='a')
        b = (a ** 2).mean()
        b.forward(a=2.5)
        b.backward()
        self.assertTrue(allclose(b.data, 6.25))
        self.assertTrue(allclose(a.grad, 5))

        b = (a ** 2).mean(axis=())
        b.forward(a=2.5)
        b.backward()
        self.assertTrue(allclose(b.data, 6.25))
        self.assertTrue(allclose(a.grad, 5))

    def test_tensordot_op(self):

        a = Value(empty((2, 3, 4)))
        b = Value(empty((4, 3, 5)))
        c = tensordot(a, b, 1)
        c.backward()
        self.assertTrue(allclose(c.shape, (2, 3, 3, 5)))

        c = tensordot(a, b, 2)
        c.backward()
        self.assertTrue(allclose(c.shape, (2, 5)))

        a = Value(empty((2,)))
        b = Value(empty((3,)))
        c = tensordot(a, b, 0)
        c.backward()
        self.assertTrue(allclose(c.shape, (2, 3)))

        # test inner product
        a = Value(array([2, 3]))
        b = Value(array([3, 4]))
        c = (a @ b) ** 2
        self.assertTrue(c.data == 18 ** 2)
        c.backward()
        self.assertTrue(allclose(a.grad, [108, 144]))
        self.assertTrue(allclose(b.grad, [72, 108]))

    def test_tensordot_op_scalar_input(self):

        a = Value(3)
        b = Value(5)
        c = tensordot(a, b, 0)
        c.backward()
        self.assertTrue(allclose(c.data, 15))
        self.assertTrue(allclose(a.grad, 5))
        self.assertTrue(allclose(b.grad, 3))

        f = Value(array([[2, 3], [4, 5]]))
        g = tensordot(a, f, 0)
        g.backward()
        self.assertTrue(allclose(g.data, array([[6, 9], [12, 15]])))
        self.assertTrue(allclose(a.grad, 14))
        self.assertTrue(allclose(f.grad, a.data))

        g = tensordot(f - 2, b, 0)
        g.backward()
        self.assertTrue(allclose(g.data, array([[0, 5], [10, 15]])))
        self.assertTrue(allclose(b.grad, 6))
        self.assertTrue(allclose(f.grad, b.data))

    def test_chain_ops(self):
        x = Value(-4.0)
        z = 2 * x + 2 + x
        q = z.relu() + z * x
        h = (z * z).relu()
        y = h + q + q * x
        y.backward()
        xmg, ymg = x, y

        self.assertTrue(isclose(xmg.data, -4))
        self.assertTrue(isclose(xmg.grad, 46))
        self.assertTrue(isclose(ymg.data, -20))
        self.assertTrue(isclose(ymg.grad, 1))

    def test_more_ops(self):

        a = Value(-4.0)
        b = Value(2.0)
        c = a + b
        d = a * b + b**3
        c += c + 1
        c += 1 + c + (-a)
        d += d * 2 + (b + a).relu()
        d += 3 * d + (b - a).relu()
        e = c - d
        f = e**2
        g = f / 2.0
        g += 10.0 / f
        g.backward()
        amg, bmg, gmg = a, b, g

        self.assertTrue(isclose(amg.data, -4))
        self.assertTrue(isclose(amg.grad, 138.83381924))
        self.assertTrue(isclose(bmg.data, 2))
        self.assertTrue(isclose(bmg.grad, 645.57725947))
        self.assertTrue(isclose(gmg.data, 24.704081632))
        self.assertTrue(isclose(gmg.grad, 1))
