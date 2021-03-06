import tf

x = tf.variable(5)
y = tf.variable(2)
loss = tf.abs(y - x)
gvs = loss.backward(tf.array2tensor(1, loss.shape))
for g, v in gvs:
    print(g, v)
train_op = tf.Optimizer(tf.const(0.01)).minimize(loss)

g = tf.Graph([x, y, train_op, loss])
for i in range(10):
    t, xx, yy, lo = g.run([train_op, x, y, loss])
    print(t, '%.2f' % xx, '%.2f' % yy, '%.2f' % lo)
    if lo <= 0.001:
        break
