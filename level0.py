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
        """
        values存储已经计算出来的值，就不用重复计算了
        :param values:
        :return:
        """
        raise Exception("not implement")

    def backward(self, grad):
        raise Exception("not implement")

    def check_type(self, other):
        if type(other) in (int, float):
            other = get_const(other)
        assert isinstance(other, Op), 'other should be Op'
        return other

    def __add__(self, other):
        other = self.check_type(other)
        return Add(self, other)

    def __sub__(self, other):
        self.check_type(other)
        return sub(self, other)

    def __mul__(self, other):
        other = self.check_type(other)
        return Mul(self, other)

    def __truediv__(self, other):
        other = self.check_type(other)
        return Div(self, other)

    def __pow__(self, p, modulo=None):
        p = self.check_type(p)
        return Power(self, p)


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
            return 'constant'
        else:
            return 'variable'


def get_const(value) -> Num:
    # 如果是可以被训练的浮点数，那么直接新建一个变量，如果是常量，那就要重复利用常量
    assert type(value) in (int, float, np.int, np.float32, np.int32), "value should be int or float,but now is %s" % (type(value))
    k = value
    if k not in constant_values:
        constant_values[k] = Num(value, NumType.constant)
    return constant_values[k]


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


def boolean_or(x, y):
    return BooleanNot(BooleanAnd(BooleanNot(x), BooleanNot(y)))


def equal(x: Op, y: Op):
    return BooleanAnd(BooleanNot(Less(x, y)), BooleanNot(Less(y, x)))


def larger(x: Op, y: Op):
    return Less(y, x)


def merge_gvs(gvs_list: list):
    # 找出那些被求导的变量，构建变量到id的映射
    vs = set()
    for gvs in gvs_list:
        for _, v in gvs:
            vs.add(v)
    v2id = dict(zip(list(vs), range(len(vs))))
    # 构建一张二维张量表，纵轴表示index，横轴表示各个变量
    a = [[get_const(0) for i in range(len(vs))] for j in range(len(gvs_list))]
    for id_, gvs in enumerate(gvs_list):
        for g, v in gvs:
            vid = v2id.get(v)
            a[id_][vid] = Add(a[id_][vid], g)
    a = np.array(a)
    return a, v2id


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


def abs(x: Op):
    return Condition(Less(x, get_const(0)), Mul(get_const(-1), x), x)


class Add(Op):
    def __init__(self, x: Op, y: Op):
        super(Add, self).__init__()
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


def sub(x: Op, y: Op) -> Add:
    # 减法=加法和乘法
    return x + y * (-1)


class Mul(Op):
    def __init__(self, x: Op, y: Op):
        super(Mul, self).__init__()
        self.x = x
        self.y = y
        self.dependency = {self.x, self.y}

    def forward(self, values):
        if not self in values:
            values[self] = self.x.forward(values) * self.y.forward(values)
        return values[self]

    def backward(self, grad):
        # 求导就是链式法则
        return self.x.backward(grad * self.y) + self.y.backward(grad * self.x)

    def __str__(self):
        return "{}*{}".format(self.x, self.y)


class Div(Op):
    def __init__(self, x: Op, y: Op):
        super(Div, self).__init__()
        self.x = x
        self.y = y
        self.dependency = {self.x, self.y}

    def forward(self, values):
        if not self in values:
            values[self] = self.x.forward(values) / self.y.forward(values)
        return values[self]

    def backward(self, grad):
        x_back = self.x.backward(grad * get_const(1) / self.y)
        y_back = self.y.backward(self.x * get_const(-1) / self.y ** 2)
        return x_back + y_back

    def __str__(self):
        return "{}/{}".format(self.x, self.y)


class Power(Op):
    def __init__(self, x: Op, y: Op):
        super(Power, self).__init__()
        self.x = x
        self.y = y
        self.dependency = [self.x, self.y]

    def forward(self, values):
        if not self in values:
            values[self] = self.x.forward(values) ** self.y.forward(values)
        return values[self]

    def backward(self, grad):
        # 求导就是链式法则
        x = self.x.backward(grad * self.y * self.x ** (self.y - 1))
        y = self.y.backward(grad * self.x ** self.y * Log(self.x))
        return x + y

    def __str__(self):
        return "%s**%s" % (self.x, self.y)


class Log(Op):
    def __init__(self, x: Op):
        super(Log, self).__init__()
        x = self.check_type(x)
        self.x = x
        self.dependency = {self.x}

    def forward(self, values):
        if not self in values:
            values[self] = np.log(self.x.forward(values))
        return values[self]

    def backward(self, grad):
        return self.x.backward(grad * get_const(1) / self.x)

    def __str__(self):
        return "log({})".format(self.x)


def exp(x):
    return Power(get_const(np.e), x)


def test1():
    x = Num(3)
    y = Num(4)
    z = x ** exp(Log(2)) + y / 3
    print(z)
    print(z.forward({}))
    a = z.backward(get_const(1.0))
    Assign(x, y).forward({})
    print(x.forward({}))
    print(y.forward({}))
    print('=' * 10)
    for grad, var in a:
        print(grad.__class__, var)


def test2():
    x = Num(2)
    y = Num(3)
    z = x / y
    print(z.forward({}))
    gv = z.backward(Num(1.0))
    print(gv)
    for g, v in gv:
        print(g, 'g')
        print(g, g.forward({}), v, type(g), type(v))


if __name__ == '__main__':
    test1()
    test2()
