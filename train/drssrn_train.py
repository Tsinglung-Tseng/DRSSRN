from symbol.drssrn_utils import DataGen, DownSampler, AlignSampler
# from incident.
from dxl.learn.model.super_resolution import SuperResolution2x, SuperResolutionBlock
from dxl.learn.model import random_crop
import tensorflow as tf
import tables
import numpy as np
import time
import functools
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# matplotlib.get_backend()
# 'module://backend_interagg'


DEFAULT_FILE = '/home/qinglong/node3share/analytical_phantom_sinogram.h5'
file = tables.open_file(DEFAULT_FILE)

class Clock:
    def clock(func):
        @functools.wraps(func)
        def clocked(*args, **kwargs):
            t0 = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - t0
            name = func.__name__
            arg_lst = []
            if args:
                arg_lst.append(', '.join(repr(arg) for arg in args))
            if kwargs:
                pairs = ['%s=%r' % (k, w) for k, w in sorted(kwargs.items())]
                arg_lst.append(', '.join(pairs))

            arg_str = ', '.join(arg_lst)
            print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, arg_str, result))
            return result

        return clocked

class FLAGS:
    class TRAIN:
        BATCH_SIZE = 32
        DOWN_SAMPLING_RATIO = 2
        SAMPLER_TARGET_SHAPE = [BATCH_SIZE, 64, 64, 1]

    class SUMMARY:
        SUMMARY_DIR = '/home/qinglong/node3share/remote_drssrn/tensorboard_log'


d = DataGen(file, FLAGS.TRAIN.BATCH_SIZE)
ds = DownSampler(d.phantom, FLAGS.TRAIN.DOWN_SAMPLING_RATIO)
aspr = AlignSampler(ds(), d.phantom, FLAGS.TRAIN.SAMPLER_TARGET_SHAPE)
train_low, train_high = aspr()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

superRe2x_ins = SuperResolution2x('sR',
                                  inputs={'input': train_low, 'label': train_high},
                                  nb_layers=2,
                                  filters=5,
                                  boundary_crop=[4, 4])
res = superRe2x_ins()
train_op = tf.train.AdamOptimizer(0.0001).minimize(res['loss'])

tf.summary.scalar("loss", res['loss'])
merged_summary = tf.summary.merge_all()

sess = tf.Session(config=config)
writer = tf.summary.FileWriter(FLAGS.SUMMARY.SUMMARY_DIR, sess.graph)

sess.run(tf.global_variables_initializer())

flag = True
counter = 0
while flag:
    try:
        _, loss_temp, summary = sess.run([train_op, res['loss'], res['inference'], merged_summary])
        writer.add_summary(summary, counter)
        counter += 1
        if counter % 100 == 0:
            print(f'Loss after {counter} batch is {loss_temp}')

    except StopIteration:
        print('done')
        flag = False












