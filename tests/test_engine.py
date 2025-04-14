from unittest import TestCase
import torch
from micrograd.engine import Value

class AutodiffTest(TestCase):

    def test_sanity_check(self):

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
