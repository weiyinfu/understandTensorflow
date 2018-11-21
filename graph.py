import threading

import numpy as np

import operations as op

"""
图模型已经呼之欲出了

在计算图的时候，需要使用threading并行加速计算
"""


def to_op_list(somthing):
    # 把张量列表，张量、Op转换成Op列表
    def tensor2list(tensor):
        return list(tensor.a.reshape(-1))

    def list2list(li):
        a = []
        for i in li:
            a.extend(to_op_list(i))
        return a

    def transform(x):
        a = None
        if type(x) == list:
            a = list2list(x)
        elif isinstance(x, op.Tensor):
            a = tensor2list(x)
        elif isinstance(x, op.Op):
            a = [x]
        else:
            assert 'unkown type %s' % (type(x))
        return a

    return transform(somthing)


def control_dependency(first_run, second_run):
    # 控制结点的依赖
    first_run = to_op_list(first_run)
    second_run = to_op_list(second_run)
    for first in first_run:
        for second in second_run:
            second.dependency.add(first)


def group(nodes):
    # 把结点打包，主要用于依赖控制
    nodes = to_op_list(nodes)
    return op.Group(nodes)


class Node:
    # node类其实就是在Op类外面包裹一层图相关的数据
    def __init__(self, id):
        self.dependency_count = 0
        self.depend_me = []
        self.op = None  # 该结点对应的操作
        self.id = id


class Worker:
    # 执行图的一个worker
    def __init__(self, thread_count=4):
        self.thread_count = thread_count
        self.q = []
        self.counter = dict()
        self.runid = 0
        self.need = set()
        self.values = dict()

    def compute(self, feed_dict: dict, need):
        self.need = need
        self.values = {k: v for k, v in feed_dict.items()}
        self.q = [node for node in self.need if node.dependency_count == 0]
        self.counter = {node: node.dependency_count for node in self.need}
        threads = []
        for i in range(self.thread_count):
            threads.append(threading.Thread(target=self.run))
            threads[-1].start()
        for i in threads:
            i.join()
        return self.values

    def run(self):
        while self.q:
            now = self.q.pop()
            now.op.forward(self.values)
            for depend_me in now.op.dependency:
                if not depend_me in self.need: continue  # 如果不需要计算，那就不用计算了
                self.counter[depend_me] -= 1
                if self.counter[depend_me] == 0:
                    self.q.append(depend_me)


def parse_feed_dict(feed_dict):
    a = dict()
    for k, v in feed_dict.items():
        oplist = to_op_list(k)
        value_list = v.reshape(-1) if isinstance(v, np.ndarray) else [v]
        for kk, vv in zip(oplist, value_list):
            a[kk] = vv
    return a


def get_value(fetches, feed_dict):
    feed_dict = parse_feed_dict(feed_dict)
    a = []
    for i in fetches:
        a.append(i.forward(feed_dict))
    return a


class Graph:
    def __init__(self, endnodes):
        self.op2node, self.nodes = self.get_graph(endnodes)
        self.check_graph(self.nodes)
        self.worker = Worker(4)

    def simplifiy(self):
        # 有向无环图是绝对可以化简的，化简之后的DAG能够极大减少运算量
        # 逐层化简有向无环图，在每层按照op的类型、操作数类型进行归类，如果相同，则合并结点
        pass

    def find_need_nodes(self, op2node, fetches):
        # 获取需要计算的结点
        q = [op2node[i] for i in to_op_list(fetches)]
        need = set(q)
        while q:
            now = q.pop()
            for i_depend in now.op.dependency:
                nex = op2node[i_depend]
                if nex not in need:
                    q.append(nex)
                    need.add(nex)
        return need

    def check_graph(self, graph: list):
        """
        拓扑排序判断图中是否包含回路，计算图应该是一个有向无环图，通过dependency属性来判断依赖
        """
        q = [node for node in graph if node.dependency_count == 0]
        handled = 0
        counter = {node: node.dependency_count for node in graph}
        while q:
            handled += 1
            now = q.pop()
            for i in now.depend_me:
                counter[i] -= 1
                if counter[i] == 0:
                    q.append(i)
        assert handled == len(graph), "found dependency ring handled=%s,len(graph)=%s" % (handled, len(graph))

    def run(self, fetches, feed_dict=None):
        if feed_dict is None: feed_dict = {}
        feed_dict = parse_feed_dict(feed_dict)
        need = self.find_need_nodes(self.op2node, fetches)
        values = self.worker.compute(feed_dict, need=need)
        a = []
        for fetch in fetches:
            a.append(fetch.forward(values))
        return a

    def get_graph(self, endnodes):
        q = []
        for nodes in endnodes:
            if isinstance(nodes, op.Tensor):
                q.extend(nodes.a.reshape(-1))
            elif isinstance(nodes, op.Op):
                q.append(nodes)
            else:
                assert False, 'unkown type %s' % (type(nodes))
        allnodes = set(q)
        while q:
            now = q.pop()
            for i in now.dependency:
                if i not in allnodes:
                    allnodes.add(i)
                    q.append(i)
        nodes = list(allnodes)
        node_id = dict(zip(nodes, range(len(nodes))))
        a = [Node(i) for i in range(len(nodes))]
        op2node = dict()
        for node, id_ in node_id.items():
            a[id_].dependency_count = len(node.dependency)
            a[id_].op = node
            op2node[node] = a[id_]
            for depend_me in node.dependency:  # 告知我的依赖，如果你好了，通知我一下
                a[node_id[depend_me]].depend_me.append(a[id_])
        return op2node, a

    def desc(self):
        print('node count', len(self.nodes))


if __name__ == '__main__':
    a = op.const(1.0)
    b = a + a
    control_dependency(a, b)
    g = Graph([b])
    g.desc()
