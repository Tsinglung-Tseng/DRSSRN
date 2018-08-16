import matplotlib.pyplot as plt
import math
import h5py
import numpy as np
import tensorflow as tf
from srcnn_utils import get_mini_batch

f = h5py.File('/home/qinglong/node3share/25k22000train','r')

# read hdf5 dataset
grp = f['/train']
grp_img = grp['img']
grp_bicubic = grp['bicubic']

# random sampling for each mini batch with seed
np.random.seed(0)
shuffled_index = list(np.random.permutation(grp_img.shape[0]))

mini_batch_size = 64
m = grp_img.shape[0]
num_complete_minibatches = math.floor(m/mini_batch_size)

input_layer = tf.placeholder(tf.float32, [None,256,256,3])
hr_img = tf.placeholder(tf.float32, [None,256,256,3])

# convolutional Layer #1
conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=64,
    kernel_size=9,
    padding="same",
    activation=tf.nn.relu)

# convolutional Layer #2
conv2 = tf.layers.conv2d(
    inputs=conv1,
    filters=32,
    kernel_size=1,
    padding="same",
    activation=tf.nn.relu)

# convolutional Layer #3
conv3 = tf.layers.conv2d(
    inputs=conv2,
    filters=3,
    kernel_size=4,
    padding="same")#,
#activation=tf.nn.relu)

# loss function
loss = tf.losses.mean_squared_error(
    labels=hr_img,
    predictions=conv3,
    weights=1.0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES)

train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

sess=tf.Session(config=config)

sess.run(tf.global_variables_initializer())

costs = []
print_cost = True
for i in range(num_complete_minibatches):
    minibatch_cost = 0

    window_start = i*mini_batch_size
    window_end = window_start + mini_batch_size

    ith_grp_img = get_mini_batch(grp_img, shuffled_index[window_start:window_end])/255
    ith_grp_bicubic = get_mini_batch(grp_bicubic, shuffled_index[window_start:window_end])/255

    _, temp_cost = sess.run([train_op, loss], feed_dict={input_layer: ith_grp_bicubic, hr_img: ith_grp_img})

    minibatch_cost += temp_cost / num_complete_minibatches

    if print_cost == True and i % 5 == 0:
        print("Cost after epoch %i: %f" % (i, minibatch_cost))
    if print_cost == True and i % 1 == 0:
        costs.append(minibatch_cost)

plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate = 0.001")
plt.show()

