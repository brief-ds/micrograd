from unittest import TestCase
from micrograd import Value, tensordot
from torch import Tensor, tensordot as torch_tensordot
from numpy import array, allclose
import numpy.random

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
        print(abs(amg.grad - apt.grad.item()))
        print(abs(bmg.grad - bpt.grad.item()))
        self.assertTrue(abs(amg.grad - apt.grad.item()) < tol)
        self.assertTrue(abs(bmg.grad - bpt.grad.item()) < tol)

    def test_unary_ops(self):

        a = Value(array([.3, .8]))
        b = a.tanh().log1p().sum()
        c = a.arctanh().log().sum()
        d = a.arcsin().sum()

        a2 = Tensor([.3, .8])
        a2.requires_grad = True
        b2 = a2.tanh().log1p().sum()
        c2 = a2.arctanh().log().sum()
        d2 = a2.arcsin().sum()

        b.backward()
        b2.backward()
        self.assertTrue(allclose(b.data, b2.data))
        self.assertTrue(allclose(a.grad, a2.grad))

        c.backward()
        a2.grad = None
        c2.backward()
        self.assertTrue(allclose(c.data, c2.data))
        self.assertTrue(allclose(a.grad, a2.grad))

        d.backward()
        a2.grad = None
        d2.backward()
        self.assertTrue(allclose(d.data, d2.data))
        self.assertTrue(allclose(a.grad, a2.grad))

    def test_tensordot(self):

        m1 = numpy.random.rand(2, 3, 4)
        m2 = numpy.random.rand(4, 3, 5)

        a = Value(m1)
        b = Value(m2)
        c = tensordot(a, b, 1).sum()

        a2 = Tensor(m1)
        a2.requires_grad = True
        b2 = Tensor(m2)
        b2.requires_grad = True
        c2 = torch_tensordot(a2, b2, 1).sum()

        c.backward()
        c2.backward()
        self.assertTrue(allclose(c.data, c2.data))
        self.assertTrue(allclose(a.grad, a2.grad))
        self.assertTrue(allclose(b.grad, b2.grad))

        c = tensordot(a, b, 2).sum()
        a2.grad = None
        b2.grad = None
        c2 = torch_tensordot(a2, b2, ([-1, -2], [0, 1])).sum()
        c.backward()
        c2.backward()
        self.assertTrue(allclose(c.data, c2.data))
        self.assertTrue(allclose(a.grad, a2.grad))
        self.assertTrue(allclose(b.grad, b2.grad))

        v1 = numpy.random.rand(2)
        v2 = numpy.random.rand(3)

        a = Value(v1)
        b = Value(v2)
        c = tensordot(a, b, 0).sum()

        a2 = Tensor(v1)
        a2.requires_grad = True
        b2 = Tensor(v2)
        b2.requires_grad = True
        c2 = torch_tensordot(a2, b2, 0).sum()

        c.backward()
        c2.backward()
        self.assertTrue(allclose(c.data, c2.data))
        self.assertTrue(allclose(a.grad, a2.grad))
        self.assertTrue(allclose(b.grad, b2.grad))

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

    def test_reduce_ops_neg_axis(self):

        a = Value(array([[[1, 2, -2], [2, 1, 0]],
                         [[-2, 1, 0], [3, 2, 1]]]))
        b = a.mean(axis=(0, -1)).relu().sum()
        c = a.mean(axis=(-3, -2)).relu().mean()

        a2 = Tensor([[[1, 2, -2], [2, 1, 0]],
                     [[-2, 1, 0], [3, 2, 1]]])
        a2.requires_grad = True
        b2 = a2.mean(axis=(0, -1)).relu().sum()
        c2 = a2.mean(axis=(-3, -2)).relu().mean()

        b.backward()
        b2.backward()
        self.assertTrue(allclose(b.data, b2.data))
        self.assertTrue(allclose(a.grad, a2.grad))

        a2.grad = None
        c.backward()
        c2.backward()
        self.assertTrue(allclose(c.data, c2.data))
        self.assertTrue(allclose(a.grad, a2.grad))
