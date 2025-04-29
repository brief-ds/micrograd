import numpy.random
from micrograd.engine import Value

class Module:

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True, values=None):
        self.w = [Value(_) for _ in values]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        values = numpy.random.uniform(-1, 1, (nin, nout))
        self.neurons = [Neuron(nin, values=v, **kwargs) for v in values.T]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

    def L2_norm(self):
        return sum([p.data ** 2 for p in self.parameters()])

    def grad_L2_norm(self):
        return sum([p.grad ** 2 for p in self.parameters()
                    if p.grad is not None])

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
