import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import level9 as tf

data = load_iris()
train_y, test_y, train_x, test_x = train_test_split(data.target, data.data)
print(train_x.shape, train_y.shape)

layer1_hidden_size = 10
place_x = tf.placeholder((1, 4))
place_y = tf.placeholder(tuple())
w = tf.variable(np.random.random((4, layer1_hidden_size)) * 0.1 - 0.05)
b = tf.variable(np.random.random((1, layer1_hidden_size,)) * 0.1 - 0.05)
wx = tf.matmul(place_x, w)
wx_b = tf.add(wx, b)
layer1 = tf.sigmoid(wx_b)
w = tf.variable(np.random.random((layer1_hidden_size, 3)))
b = tf.variable(np.random.random((1, 3)))
wx = tf.matmul(layer1, w)
wx_b = tf.add(wx, b)
wx_b_flat = tf.reshape(wx_b, (-1,))
logits = tf.softmax(wx_b_flat)

argmax = tf.Tensor(np.array(tf.argmax(logits)))
eq = tf.equal(argmax, place_y)
accuracy = eq
loss = tf.sparse_cross_entropy(logits, place_y)

learn_rate = tf.const(0.001)
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
