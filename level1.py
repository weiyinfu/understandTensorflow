import numpy as np

import level0

"""
张量就是一堆Operation摆成特定形状

数和张量谁是本源？
数可以说是一维长度为1的张量
张量可以说是若干维的数字

谁简单、谁粒度更小，谁就是本源
"""


def each(a, func):
    return np.reshape([func(i) for i in np.reshape(a, -1)], a.shape)


def each_many(array_tuple, func):
    a = [i.a.reshape(-1) for i in array_tuple]
    return np.reshape([func(*i) for i in zip(*a)], array_tuple[0].shape)


class Tensor(level0.Op):
    def __init__(self, a):
        super(Tensor, self).__init__()
        self.a = a
        self.shape = self.a.shape

    def forward(self, values):
        return each(self.a, lambda x: x.forward(values))

    def backward(self, grad):
        grad = self.check_type(grad)
        a = []
        for g, v in zip(grad.a.reshape(-1), self.a.reshape(-1)):
            a += v.backward(g)
        return a

    def check_type(self, other):
        if type(other) in (int, float):
            other = array2tensor(other, self.shape, False)
        elif isinstance(other, Tensor):
            pass
        elif isinstance(other, level0.Op):
            other = Tensor(np.broadcast_to(other, self.shape))
        else:
            assert False, 'cannot accept type %s' % (type(other))
        return other

    def __getitem__(self, item):
        return self.a[item]

    def __setitem__(self, key, value):
        self.a[key] = value

    def __add__(self, other):
        other = self.check_type(other)
        return add(self, other)

    def __sub__(self, other):
        other = self.check_type(other)
        return sub(self, other)

    def __mul__(self, other):
        other = self.check_type(other)
        return mul(self, other)

    def __truediv__(self, other):
        other = self.check_type(other)
        return div(self, other)

    def __pow__(self, po, modulo=None):
        po = self.check_type(po)
        return power(self, po)

    def __matmul__(self, other):
        other = self.check_type(other)
        return matmul(self, other)


def unary_op(x, op):  # 一元运算符
    return Tensor(each(x.a, op))


def binary_op(x: Tensor, y: Tensor, binary):
    assert x.shape == y.shape, 'x.shape and y.shape should match,x.shape=%s,y.shape=%s'.format(x.shape, y.shape)
    return Tensor(each_many((x, y), binary))


def tri_op(x: Tensor, y: Tensor, z: Tensor, tri):
    assert x.shape == y.shape == z.shape, 'x.shape and y.shape and z.shape should match'
    return Tensor(each((x, y, z), tri))


def condition(x: Tensor, y: Tensor, z: Tensor):
    return tri_op(x, y, z, level0.Condition)


def abs(x: Tensor):
    return unary_op(x, level0.abs)


def equal(x: Tensor, y: Tensor):
    return binary_op(x, y, level0.equal)


def add(x: Tensor, y: Tensor):
    return binary_op(x, y, level0.Add)


def sub(x: Tensor, y: Tensor):
    return binary_op(x, y, level0.sub)


def mul(x: Tensor, y: Tensor):
    return binary_op(x, y, level0.Mul)


def div(x: Tensor, y: Tensor):
    return binary_op(x, y, level0.Div)


def power(x: Tensor, y: Tensor):
    return binary_op(x, y, level0.Power)


def log(x: Tensor):
    return unary_op(x, level0.Log)


def exp(x: Tensor):
    return unary_op(x, level0.exp)


def assign(src: Tensor, target: Tensor):
    return binary_op(src, target, level0.Assign)


def matmul(x: Tensor, y: Tensor):
    assert len(x.shape) == 2 and len(y.shape) == 2 and x.shape[1] == y.shape[0], 'now can only matmul two matrix'
    a = np.empty((x.shape[0], y.shape[1]), dtype=object)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a[i][j] = level0.get_const(0)
            for k in range(x.shape[1]):
                a[i][j] = x[i, k] * y[k, j] + a[i][j]
    return Tensor(a)


def array2tensor(a, shape=None, trainable=True):
    # 把numpy数组转换成tensor
    a = np.array(a, dtype=np.float32)
    variable_converter = lambda x: level0.Num(x)
    const_converter = lambda x: level0.get_const(x)
    converter = variable_converter if trainable else const_converter
    if shape is None:
        return Tensor(each(a, converter))
    return Tensor(each(np.broadcast_to(a, shape), converter))


def reshape(a: Tensor, shape):
    return Tensor(np.reshape(a.a, shape))


def test1():
    a = array2tensor(np.random.randint(1, 5, (3, 3)), trainable=True)
    b = exp(log(array2tensor(np.random.randint(1, 5, (3, 3)), trainable=False)))
    print(a.forward({}))
    print(b.forward({}))
    z = a @ b
    print(z[0, 0])
    print(z.forward({}))
    x = z.backward(array2tensor(1.0, z.shape))
    print(len(x), type(x))
    for i in x:
        print(i[1])
    print('--' * 10)
    assign(b, a).forward({})
    print(b.forward({}))


def test2():
    a = array2tensor(np.array(3))
    b = array2tensor(np.array(4))
    c = a * b
    print(c.forward({}))
    gv = c.backward(array2tensor(np.array(1.0), c.shape))
    for g, v in gv:
        print(g, v, type(g), type(v))
        print(g.forward({}))


if __name__ == '__main__':
    a = array2tensor([1, -2, -3])
    print(abs(a).forward({}))
