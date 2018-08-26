# import numpy as np
# import matplotlib
# # matplotlib.use('Agg')
# from matplotlib import pyplot as plt
#
# print(matplotlib.get_backend())
#
# random_image = np.random.random([500, 500])
# print(random_image)
# plt.imshow(random_image, cmap='gray')
# plt.colorbar()
# plt.show()

from symbol.drssrn_utils import DataGen, DownSampler, AlignSampler
import tensorflow as tf

DEFAULT_FILE = '/home/qinglong/node3share/analytical_phantom_sinogram.h5'
file = tables.open_file(DEFAULT_FILE)

class FLAGS:
    class TRAIN:
        BATCH_SIZE = 32
        DOWN_SAMPLING_RATIO = 2
        SAMPLER_TARGET_SHAPE = [BATCH_SIZE, 64, 64, 1]

    class SUMMARY:
        SUMMARY_DIR = '/home/qinglong/node3share/remote_drssrn/tensorboard_log'

with tf.device('/device:GPU:1'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

    d = DataGen(file)

# Creates a session with log_device_placement set to True.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

sess = tf.Session(config=config)
# Runs the op.
print(sess.run(c))