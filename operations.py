import numpy as np

"""
数是万事万物的本源
基本操作都是作用于数字的，而不是作用于张量的
只有一种数据类型：float，一切皆float
"""

constant_values = dict()  # 常量池


class Op:
    def __init__(self):
        self.dependency = set()

    def forward(self, values):
        raise Exception("not implement")

    def backward(self, grad):
        raise Exception("not implement")


class Group(Op):
    # 只用来把结点打包，不做任何操作
    def __init__(self, depend_list):
        super(Group, self).__init__()
        for i in depend_list:
            self.dependency.add(i)

    def forward(self, values):
        for i in self.dependency:
            i.forward(values)
        return None

    def backward(self, grad):
        return []

    def __str__(self):
        return ','.join(str(i) for i in self.dependency)


class NumType:
    variable = 1
    placeholder = 2
    constant = 3


class Num(Op):
    def __init__(self, value, numtype=NumType.variable):
        super(Num, self).__init__()
        self.value = value
        self.numtype = numtype

    def forward(self, values):
        if self.numtype == NumType.placeholder:
            assert self in values, 'no feed data for placeholder'
            return values[self]
        else:
            return self.value

    def backward(self, grad):
        # 返回grad and vars
        if self.numtype == NumType.variable:
            return [(grad, self)]
        else:
            return []

    def __str__(self):
        if self.numtype == NumType.placeholder:
            return 'place'
        elif self.numtype == NumType.constant:
            return str(self.value)
        else:
            return 'variable'


def const_num(value) -> Num:
    # 如果是可以被训练的浮点数，那么直接新建一个变量，如果是常量，那就要重复利用常量
    assert type(value) in (int, float, np.int, np.float32, np.int32), "value should be int or float,but now is %s" % (type(value))
    k = value
    if k not in constant_values:
        constant_values[k] = Num(value, NumType.constant)
    return constant_values[k]


def merge_gvs(gvs_list: list):
    # 找出那些被求导的变量，构建变量到id的映射
    vs = set()
    for gvs in gvs_list:
        for _, v in gvs:
            vs.add(v)
    v2id = dict(zip(list(vs), range(len(vs))))
    # 构建一张二维张量表，纵轴表示index，横轴表示各个变量
    a = [[const_num(0) for i in range(len(vs))] for j in range(len(gvs_list))]
    for id_, gvs in enumerate(gvs_list):
        for g, v in gvs:
            vid = v2id.get(v)
            a[id_][vid] = AddOp(a[id_][vid], g)
    a = np.array(a)
    return a, v2id


class Assign(Op):
    # 赋值操作
    def __init__(self, src: Op, target: Num):
        super(Assign, self).__init__()
        assert type(target) == Num, 'can only assign Num to Num，but target is %s' % (type(target))
        self.dependency = {target}
        self.src = src
        self.target = target

    def forward(self, values):
        self.target.value = self.src.forward(values)
        return None

    def backward(self, grad):
        return []

    def __str__(self):
        return "%s=%s" % (self.target, self.src)


class Less(Op):
    def __init__(self, x, y):
        super(Less, self).__init__()
        self.x = x
        self.y = y
        self.dependency = {x, y}

    def forward(self, values):
        x_value = self.x.forward(values)
        y_value = self.y.forward(values)
        return 1 if x_value < y_value else 0

    def backward(self, grad):
        return []

    def __str__(self):
        return "%s<%s" % (self.x, self.y)


class BooleanNot(Op):
    def __init__(self, x):
        super(BooleanNot, self).__init__()
        self.x = x
        self.dependency = {x}

    def forward(self, values):
        x_value = self.x.forward(values)
        return 1 - int(bool(x_value))

    def backward(self, grad):
        return []


class BooleanAnd(Op):
    def __init__(self, x: Op, y: Op):
        super(BooleanAnd, self).__init__()
        self.x = x
        self.y = y
        self.dependency = {x, y}

    def forward(self, values):
        return self.x.forward(values) and self.y.forward(values)

    def backward(self, grad):
        return []


class Condition(Op):
    def __init__(self, condition: Op, if_true: Op, if_false: Op):
        super(Condition, self).__init__()
        self.dependency = [condition, if_true, if_false]
        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false

    def forward(self, values):
        res = self.condition.forward(values)
        if res:
            return self.if_true.forward(values)
        else:
            return self.if_false.forward(values)

    def backward(self, grad):
        if_true = self.if_true.backward(grad)
        if_false = self.if_false.backward(grad)
        a, v2id = merge_gvs([if_true, if_false])
        gvs = [(Condition(self.condition, a[0, vid], a[1, vid]), v) for v, vid in v2id.items()]
        return gvs

    def __repr__(self):
        return "if(%s)then %s else %s" % (self.condition, self.if_true, self.if_false)


class AddOp(Op):
    def __init__(self, x: Op, y: Op):
        super(AddOp, self).__init__()
        self.x = x
        self.y = y
        self.dependency = {self.x, self.y}

    def forward(self, values):
        if not self in values:
            values[self] = self.x.forward(values) + self.y.forward(values)
        return values[self]

    def backward(self, grad):
        # 求导就是链式法则
        return self.x.backward(grad) + self.y.backward(grad)

    def __str__(self):
        return "({}+{})".format(self.x, self.y)


class MulOp(Op):
    def __init__(self, x: Op, y: Op):
        super(MulOp, self).__init__()
        self.x = x
        self.y = y
        self.dependency = {self.x, self.y}

    def forward(self, values):
        if not self in values:
            values[self] = self.x.forward(values) * self.y.forward(values)
        return values[self]

    def backward(self, grad):
        # 求导就是链式法则
        return self.x.backward(MulOp(grad, self.y)) + self.y.backward(MulOp(grad, self.x))

    def __str__(self):
        return "{}*{}".format(self.x, self.y)


class DivOp(Op):
    def __init__(self, x: Op, y: Op):
        super(DivOp, self).__init__()
        self.x = x
        self.y = y
        self.dependency = {self.x, self.y}

    def forward(self, values):
        if not self in values:
            values[self] = self.x.forward(values) / self.y.forward(values)
        return values[self]

    def backward(self, grad):
        x_back = self.x.backward(DivOp(MulOp(grad, const_num(1)), self.y))
        y_back = self.y.backward(MulOp(self.x, DivOp(const_num(-1), MulOp(self.y, self.y))))
        return x_back + y_back

    def __str__(self):
        return "{}/{}".format(self.x, self.y)


class PowerOp(Op):
    def __init__(self, x: Op, y: Op):
        super(PowerOp, self).__init__()
        self.x = x
        self.y = y
        self.dependency = [self.x, self.y]

    def forward(self, values):
        if not self in values:
            values[self] = self.x.forward(values) ** self.y.forward(values)
        return values[self]

    def backward(self, grad):
        # 求导就是链式法则
        x = self.x.backward(MulOp(grad, MulOp(self.y, PowerOp(self.x, AddOp(self.y, const_num(-1))))))
        y = self.y.backward(MulOp(grad, PowerOp(self.x, MulOp(self.y, Log(self.x)))))
        return x + y

    def __str__(self):
        return "%s**%s" % (self.x, self.y)


class Log(Op):
    def __init__(self, x: Op):
        super(Log, self).__init__()
        self.x = x
        self.dependency = {self.x}

    def forward(self, values):
        if not self in values:
            values[self] = np.log(self.x.forward(values))
        return values[self]

    def backward(self, grad):
        return self.x.backward(MulOp(grad, DivOp(const_num(1), self.x)))

    def __str__(self):
        return "log({})".format(self.x)


def each(array_tuple, func):
    a = [i.reshape(-1) for i in array_tuple]
    return np.reshape([func(*i) for i in zip(*a)], array_tuple[0].shape)


class Tensor(Op):
    def __init__(self, a):
        super(Tensor, self).__init__()
        self.a = a
        self.shape = self.a.shape

    def forward(self, values):
        return each((self.a,), lambda x: x.forward(values))

    def backward(self, grad):
        if not isinstance(grad, Tensor):
            grad = Tensor(np.array(grad))
        assert grad.shape == self.shape, 'self.shape %s ;grad.shape=%s' % (self.shape, grad.shape)
        a = []
        for g, v in zip(grad.a.reshape(-1), self.a.reshape(-1)):
            a += v.backward(g)
        return a

    def reshape(self, shape):
        a = self.a.reshape(shape)
        return Tensor(a)

    def __getitem__(self, item):
        return self.a[item]

    def __setitem__(self, key, value):
        self.a[key] = value

    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __matmul__(self, other):
        return matmul(self, other)

    def __le__(self, other):
        return less(self, other)

    def __truediv__(self, other):
        return div(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __pow__(self, p, modulo=None):
        return power(self, p)


class Argmax(Op):
    def __init__(self, x: Tensor):
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


class Index(Op):
    """
    获取tensor的某个下标
    下标如何求导，这真是一个大难题呀
    只返回一个数值
    """

    def __init__(self, tensor: Tensor, index: Tensor):
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
        a, v2id = merge_gvs(gvs_list)
        gvs = [(Index(Tensor(a[:, ind]), self.index), v) for v, ind in v2id.items()]
        return gvs

    def __str__(self):
        return "%s[%s]" % (self.tensor, self.index)


def check_type(*op_list):
    tensor_shape = None
    for op in op_list:
        if isinstance(op, Tensor):
            if tensor_shape is None or len(op.shape) > len(tensor_shape):
                tensor_shape = op.shape
    if tensor_shape is None:  # 构建一个模板
        tensor_shape = tuple()
    a = []
    for op in op_list:
        if not isinstance(op, Tensor) or op.shape != tensor_shape:
            if type(op) in (int, float, np.float32, np.float64, np.float):
                op = const_num(op)
            assert isinstance(op, Op), 'op should be Op,but got %s' % (type(op))
            op = np.broadcast_to(np.array(op), tensor_shape)
        a.append(op)
    return tuple(list(a))


def add(x, y):
    x, y = check_type(x, y)
    return Tensor(each((x, y), AddOp))


def mul(x, y):
    x, y = check_type(x, y)
    return Tensor(each((x, y), MulOp))


def div(x, y):
    x, y = check_type(x, y)
    return Tensor(each((x, y), DivOp))


def assign(src, target):
    src, target = check_type(src, target)
    return Tensor(each((src, target), Assign))


def boolean_or(x, y):
    x, y = check_type(x, y)
    one = lambda x, y: BooleanNot(BooleanAnd(BooleanNot(x), BooleanNot(y)))
    return Tensor(each((x, y), one))


def equal(x: Op, y: Op):
    one = lambda x, y: BooleanAnd(BooleanNot(Less(x, y)), BooleanNot(Less(y, x)))
    x, y = check_type(x, y)
    return Tensor(each((x, y), one))


def less(x: Op, y: Op):
    one = lambda x, y: Less(x, y)
    x, y = check_type(x, y)
    return Tensor(each((x, y), one))


def larger(x: Op, y: Op):
    return Tensor(less(y, x))


def abs(x: Op):
    one = lambda x: Condition(Less(x, const_num(0)), MulOp(const_num(-1), x), x)
    x, = check_type(x)
    return Tensor(each((x,), one))


def sub(x: Op, y: Op) -> AddOp:
    # 减法=加法和乘法
    one = lambda x, y: AddOp(x, MulOp(const_num(-1), y))
    x, y = check_type(x, y)
    return Tensor(each((x, y), one))


def matmul(x: Tensor, y: Tensor):
    assert len(x.shape) == 2 and len(y.shape) == 2 and x.shape[1] == y.shape[0], 'now can only matmul two matrix'
    a = np.empty((x.shape[0], y.shape[1]), dtype=object)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a[i][j] = const_num(0)
            for k in range(x.shape[1]):
                a[i][j] = AddOp(MulOp(x[i, k], y[k, j]), a[i][j])
    return Tensor(a)


def array2tensor(a, shape=None, trainable=True):
    # 把numpy数组转换成tensor
    a = np.array(a, dtype=np.float32)
    converter = Num if trainable else const_num
    if shape is None:
        return Tensor(each((a,), converter))
    return Tensor(each((np.broadcast_to(a, shape),), converter))


def reshape(a: Tensor, shape):
    return Tensor(np.reshape(a.a, shape))


def exp(x):
    return power(const_num(np.e), x)


def log(x):
    x, = check_type(x)
    return Tensor(each((x,), Log))


def power(x, y):
    x, y = check_type(x, y)
    return Tensor(each((x, y), PowerOp))


def placeholder(shape):
    return Tensor(np.broadcast_to(Num(0, NumType.placeholder), shape))


def variable(init_value, trainable=True):
    return array2tensor(init_value, trainable=trainable)


def const(value):
    a = array2tensor(value, trainable=False)
    for i in a.a.reshape(-1):
        i.num_type = NumType.constant
    return a


if __name__ == '__main__':
    x = variable(2)
    y = variable(-3)
    z = abs(x) * abs(y)
    print(z.forward({}))
    for g, v in z.backward(const(np.ones(z.shape))):
        print(g, v, g.forward({}))
