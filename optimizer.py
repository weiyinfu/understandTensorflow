import numpy as np

import operations as op


class Optimizer:
    def __init__(self, learn_rate):
        self.learn_rate = learn_rate

    def merge(self, grad_vars):
        a = dict()
        for g, v in grad_vars:
            if v not in a:
                a[v] = g
            else:
                a[v] = op.AddOp(a[v], g)
        return a

    def compute(self, loss: op.Tensor):
        if isinstance(loss, op.Tensor):
            loss = np.reshape(loss.a, -1)
        else:
            loss = [loss]
        grad_vars = []
        for lo in loss:
            grad_vars.extend(lo.backward(op.const(1.0)))
        return self.merge(grad_vars)

    def apply(self, grad_vars):
        a = []
        for v, g in grad_vars.items():
            a.append(op.assign(op.sub(v, op.mul(g, self.learn_rate)), v))
        return op.Group(a)

    def minimize(self, loss):
        grad_vars = self.compute(loss)
        return self.apply(grad_vars)
