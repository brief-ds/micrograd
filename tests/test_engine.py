from unittest import TestCase
from micrograd.engine import Value
from numpy import array, isclose, allclose, nan

class AutodiffTest(TestCase):

    def test_basic(self):

        a = Value(shape=(2,), name='a')
        b = Value(shape=(2,), name='b')
        c = (a + 2).relu() * b ** 2
        c.forward(a=array([-3, 1]), b=array([2, 3]))
        c.backward()
        self.assertTrue(allclose(c.data, [0, 27]))
        self.assertTrue(allclose(a.grad, [0, 9]))
        self.assertTrue(allclose(b.grad, [0, 18]))
        self.assertTrue(allclose(c.grad, [1, 1]))

        d = a.log1p()
        d.forward(a=array([2, 3]))
        d.backward()
        self.assertTrue(allclose(d.data, [1.09861229, 1.38629436]))
        self.assertTrue(allclose(a.grad, [0.33333333, 0.25]))

        d.forward(a=array([-2, 3]))
        d.backward()
        self.assertTrue(allclose(d.data, [nan, 1.38629436], equal_nan=True))
        self.assertTrue(allclose(a.grad, [nan, 0.25], equal_nan=True))

    def test_sanity_check(self):

        a = Value(shape=(2,), name='a')
        a.forward(a=array([2, 3]))
        a.backward()
        self.assertTrue(allclose(a.grad, [1, 1]))

    def test_ops(self):
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
