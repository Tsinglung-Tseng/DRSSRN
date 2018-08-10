import tensorflow as tf
import numpy as np
#from dxl.learn.model import random_crop

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

# x1 = tf.Variable(tf.random_normal([1,3,4,4,32], stddev=0.35), name="weights")
# x2 = tf.Variable(tf.random_normal([1,3,4,4,64], stddev=0.35), name="weights")
# x3 = tf.Variable(tf.random_normal([1,3,4,4,64], stddev=0.35), name="weights")
#
# with tf.Session(config=config) as test:
#     x = tf.concat(
#         [x1,x2,x3],
#         axis=-1
#     )
#     print(np.shape(x))

from dxl.learn.model.crop import random_crop
from keras.layers import Conv2D
import keras.backend as K


sess = tf.Session(config = config)
K.set_session(sess)

xxx_ph = tf.placeholder('float', [3,6,6,10])
xxx = np.random.randn(3,6,6,10)

X = Conv2D(filters=32,strides=(1,1), kernel_size=(3,3))(xxx_ph)
#X = tf.layers.conv2d(inputs=xxx_ph, kernel_size=(3,3), filters=32, strides=(1,1))

sess.run(tf.initialize_all_variables())
yo = sess.run(X, feed_dict={xxx_ph: xxx})

import matplotlib.pyplot as plt
plt.plot(yo[1][3][3], yo[1][3][2])
plt.show()