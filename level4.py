import numpy as np

import level0
import level1
import level3

"""
更高级的封装，更牛逼的函数
有些函数看似不能求导，实际上是可以求导的
"""


class Optimizer:
    def __init__(self, learn_rate):
        assert type(learn_rate) == level0.Num, "learn_rate should be a Num"
        self.learn_rate = learn_rate

    def merge(self, grad_vars):
        a = dict()
        for g, v in grad_vars:
            if v not in a:
                a[v] = g
            else:
                a[v] = level0.Add(a[v], g)
        return a

    def compute(self, loss: level1.Tensor):
        if isinstance(loss, level1.Tensor):
            loss = np.reshape(loss.a, -1)
        else:
            loss = [loss]
        grad_vars = []
        for lo in loss:
            grad_vars.extend(lo.backward(level0.get_const(1.0)))
        return self.merge(grad_vars)

    def apply(self, grad_vars):
        a = []
        for v, g in grad_vars.items():
            a.append(level0.Assign(level0.sub(v, g * self.learn_rate), v))
        return level3.group(a)

    def minimize(self, loss):
        grad_vars = self.compute(loss)
        return self.apply(grad_vars)


def placeholder(shape):
    """
    为None的维度就直接当成是1
    然后对于输入的多个维度，依次填入求值，返回的形状与输入的形状相同
    :param shape:
    :return:
    """
    return level1.Tensor(np.broadcast_to(level0.Num(0, level0.NumType.placeholder), shape))


def variable(init_value, trainable=True):
    return level1.array2tensor(init_value, trainable=trainable)


class Argmax(level0.Op):
    def __init__(self, x: level1.Tensor):
        super(Argmax, self).__init__()
        self.x = x
        self.dependency = list(np.reshape(x, -1))

    def forward(self, values):
        a = self.x.forward(values)
        return np.argmax(a)

    def backward(self, grad):
        return []

    def __str__(self):
        return "argmax(%s)" % (self.x)


class Index(level0.Op):
    """
    获取tensor的某个下标
    下标如何求导，这真是一个大难题呀
    只返回一个数值
    """

    def __init__(self, tensor: level1.Tensor, index: level1.Tensor):
        super(Index, self).__init__()
        assert len(tensor.shape) == 1, 'index need a vector ,but now tensor.shape is {}'.format(tensor.shape)
        assert index.shape == tuple(), 'index must be value'
        self.tensor = tensor
        self.index = index
        self.dependency = list(np.reshape(tensor.a, -1)) + list(np.reshape(index.a, -1))

    def forward(self, values):
        ind = int(self.index.forward(values))
        value = self.tensor.a[ind].forward(values)
        return value

    def backward(self, grad):
        # 先让各个分量求导
        gvs_list = [x.backward(grad) for x in np.reshape(self.tensor.a, -1)]
        a, v2id = level0.merge_gvs(gvs_list)
        gvs = [(Index(level1.Tensor(a[:, ind]), self.index), v) for v, ind in v2id.items()]
        return gvs

    def __str__(self):
        return "%s[%s]" % (self.tensor, self.index)


def sparse_cross_entropy(logits: level1.Tensor, labels: level1.Tensor):
    assert len(logits.shape) == 1, 'but now is %s' % logits.shape
    assert labels.shape == tuple(), 'label.shape should be a number'
    logits = level1.log(logits)
    return level0.get_const(-1) * Index(logits, labels)


def test1():
    a = level1.array2tensor([5, 2, 3, 4, 5])
    print(a.forward({}))
    i = a * Index(a, level1.Tensor(np.array(level0.get_const(3))))
    print('i.forward()', i.forward({}), i.shape, type(i))
    t = level1.array2tensor(1.0, i.shape)
    gv = i.backward(t)
    print(gv)
    for g, v in gv:
        print(g.forward({}), v.forward({}), 'gv')


def test2():
    a = level1.array2tensor([1, 2, 3, 4, 5])
    print(sparse_cross_entropy(a, level1.Tensor(np.array(level0.get_const(2)))).forward({}))
    print(np.log(a.forward({})))


if __name__ == '__main__':
    # test1()
    test2()
