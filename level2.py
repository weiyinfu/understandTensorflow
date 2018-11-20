"""
可以在tensor基本运算的基础上实现一些复杂函数了
"""

import numpy as np
import level0
import level1


def reduce_sum(x: level1.Tensor):
    assert len(x.shape) == 1, 'reduce sum only accept vector'
    s = level0.get_const(0)
    for i in x.a:
        s = level0.Add(s, i)
    return level1.Tensor(np.array(s))


def softmax(x: level1.Tensor):
    assert len(x.a.shape) == 1, 'softmax only accept vector'
    y = level1.exp(x)
    s = reduce_sum(y)
    s = level1.Tensor(np.broadcast_to(s, y.shape))
    y = y / s
    return y


def sigmoid(x):
    up = level1.Tensor(np.broadcast_to(level0.get_const(1.0), x.shape))  # 分子
    return up / (up + x)


def reduce_mean(x):
    return reduce_sum(x) / level1.Tensor(np.array(level0.get_const(x.shape[0])))


def cross_entropy(logits: level1.Tensor, label: level1.Tensor):
    assert logits.shape == label.shape
    return level0.get_const(-1) * reduce_sum(level1.log(logits) * label)


def softmax_cross_entropy(logits: level1.Tensor, label: level1.Tensor):
    assert logits.shape == label.shape
    return softmax(cross_entropy(logits, label))


if __name__ == '__main__':
    print(cross_entropy(level1.array2tensor([1, 2, 3, 4]), level1.array2tensor([5, 6, 7, 8])).forward({}))
