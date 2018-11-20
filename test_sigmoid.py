import pylab as plt
import level9 as tf
import numpy as np

x = tf.variable(np.linspace(-15, 15, 100))
fig, (sigmoid, abs) = plt.subplots(1, 2)
sigmoid.plot(x.forward({}), tf.sigmoid(x).forward({}), label='sigmoid')
sigmoid.set_title('sigmoid')
abs.plot(x.forward({}), tf.abs(x).forward({}), label='abs')
abs.set_title('abs')
plt.show()
