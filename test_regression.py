import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import level9 as tf

"""
线性回归的效果比多层神经网络效果好
"""


def rand(shape):
    return np.random.random(shape) * 0.01 - 0.005


def activate(x):
    ones = tf.Tensor(np.broadcast_to(tf.const(1), x.shape))
    return ones / (ones + x)


data = load_iris()
train_y, test_y, train_x, test_x = train_test_split(data.target, data.data)
print(train_x.shape, train_y.shape)
place_x = tf.placeholder((1, 4))
place_y = tf.placeholder(tuple())
w = tf.variable(rand((4, 3)))
b = tf.variable(rand((1, 3)))
wx_b = tf.matmul(place_x, w) + b
wx_b_flat = tf.reshape(wx_b, (-1,))
logits = tf.softmax(wx_b_flat)
argmax = tf.Tensor(np.array(tf.argmax(logits)))
accuracy = tf.equal(argmax, place_y)
loss = tf.sparse_cross_entropy(logits, place_y)
learn_rate = tf.const(0.01)
train_op = tf.Optimizer(learn_rate).minimize(loss)
g = tf.Graph([train_op, loss, accuracy])
g.desc()
for epoch in range(10):
    print(epoch, '=' * 20)
    for batch_x, batch_y in zip(train_x, train_y):
        while 1:
            # 东一榔头西一棒槌学不好，必须要先把一个学会再学另外一个
            _, acc, lo = g.run([train_op, accuracy, loss], feed_dict={
                place_x: batch_x,
                place_y: batch_y,
            })
            print(acc, lo)
            if acc == 1:
                break
print('train over', '-' * 10)


def test(test_x, test_y):
    total_acc = 0
    total_count = 0
    for batch_x, batch_y in zip(test_x, test_y):
        [acc] = g.run([accuracy], feed_dict={
            place_x: batch_x,
            place_y: batch_y,
        })
        total_acc += acc
        total_count += 1
    print(total_acc / total_count)


print('train data set accuracy')
test(train_x, train_y)
print('test data set accuracy')
test(test_x, test_y)
