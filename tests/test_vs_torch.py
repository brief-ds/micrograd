from unittest import TestCase
from micrograd import Value
from torch import Tensor
from numpy import array, allclose

class AutodiffTest(TestCase):

    def test_sanity_check(self):

        x = Value(-4.0)
        z = 2 * x + 2 + x
        q = z.relu() + z * x
        h = (z * z).relu()
        y = h + q + q * x
        y.backward()
        xmg, ymg = x, y

        x = Tensor([-4.0]).double()
        x.requires_grad = True
        z = 2 * x + 2 + x
        q = z.relu() + z * x
        h = (z * z).relu()
        y = h + q + q * x
        y.backward()
        xpt, ypt = x, y

        # forward pass went well
        self.assertTrue(ymg.data == ypt.data.item())
        # backward pass went well
        self.assertTrue(xmg.grad == xpt.grad.item())

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

        a = Tensor([-4.0]).double()
        b = Tensor([2.0]).double()
        a.requires_grad = True
        b.requires_grad = True
        c = a + b
        d = a * b + b**3
        c = c + c + 1
        c = c + 1 + c + (-a)
        d = d + d * 2 + (b + a).relu()
        d = d + 3 * d + (b - a).relu()
        e = c - d
        f = e**2
        g = f / 2.0
        g = g + 10.0 / f
        g.backward()
        apt, bpt, gpt = a, b, g

        tol = 1e-6
        # forward pass went well
        self.assertTrue(abs(gmg.data - gpt.data.item()) < tol)
        # backward pass went well
        self.assertTrue(abs(amg.grad - apt.grad.item()) < tol)
        self.assertTrue(abs(bmg.grad - bpt.grad.item()) < tol)

    def test_unary_ops(self):

        a = Value(array([.3, .8]))
        b = a.tanh().log1p().sum()
        c = a.arctanh().log().sum()

        a2 = Tensor([.3, .8])
        a2.requires_grad = True
        b2 = a2.tanh().log1p().sum()
        c2 = a2.arctanh().log().sum()

        b.backward()
        b2.backward()
        self.assertTrue(allclose(b.data, b2.data))
        self.assertTrue(allclose(a.grad, a2.grad))

        c.backward()
        a2.grad = None
        c2.backward()
        self.assertTrue(allclose(c.data, c2.data))
        self.assertTrue(allclose(a.grad, a2.grad))

    def test_tensordot(self):
        pass

    def test_reduce_ops(self):

        a = Value(array([[[1, 2, -2], [2, 1, 0]],
                         [[-2, 1, 0], [3, 2, 1]]]))
        b = a.mean(axis=(0, 2)).relu().sum()
        c = a.mean(axis=(0, 1)).relu().mean()

        a2 = Tensor([[[1, 2, -2], [2, 1, 0]],
                     [[-2, 1, 0], [3, 2, 1]]])
        a2.requires_grad = True
        b2 = a2.mean(axis=(0, 2)).relu().sum()
        c2 = a2.mean(axis=(0, 1)).relu().mean()

        b.backward()
        b2.backward()
        self.assertTrue(allclose(b.data, b2.data))
        self.assertTrue(allclose(a.grad, a2.grad))

        a2.grad = None
        c.backward()
        c2.backward()
        self.assertTrue(allclose(c.data, c2.data))
        self.assertTrue(allclose(a.grad, a2.grad))
