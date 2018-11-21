"""
可以在tensor基本运算的基础上实现一些复杂函数了
"""

import numpy as np

import operations as op


def reduce_sum(x: op.Tensor):
    assert len(x.shape) == 1, 'reduce sum only accept vector'
    s = op.const(0)
    for i in x.a:
        s = op.AddOp(s, i)
    return op.Tensor(np.array(s))


def softmax(x: op.Tensor):
    assert len(x.a.shape) == 1, 'softmax only accept vector'
    y = op.exp(x)
    s = reduce_sum(y)
    s = op.Tensor(np.broadcast_to(s, y.shape))
    y = y / s
    return y


def sigmoid(x):
    return op.const(1) / (op.const(1) + op.exp(op.const(-1) * x))


def sparse_cross_entropy(logits: op.Tensor, labels: op.Tensor):
    assert len(logits.shape) == 1, 'but now is %s' % logits.shape
    assert labels.shape == tuple(), 'label.shape should be a number'
    logits = op.log(logits)
    return op.const(-1) * op.Index(logits, labels)


def reduce_mean(x):
    return reduce_sum(x) / op.Tensor(np.array(op.const(x.shape[0])))


def cross_entropy(logits: op.Tensor, label: op.Tensor):
    assert logits.shape == label.shape
    return op.const(-1) * reduce_sum(op.log(logits) * label)


def softmax_cross_entropy(logits: op.Tensor, label: op.Tensor):
    assert logits.shape == label.shape
    return softmax(cross_entropy(logits, label))


if __name__ == '__main__':
    print(cross_entropy(op.array2tensor([1, 2, 3, 4]), op.array2tensor([5, 6, 7, 8])).forward({}))
