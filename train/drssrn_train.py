from symbol.drssrn_utils import DataGen, DownSampler, AlignSampler
# from incident.
from dxl.learn.model.super_resolution import SuperResolution2x, SuperResolutionBlock
from dxl.learn.model import random_crop
import matplotlib.pyplot as plt
import tensorflow as tf
import tables
import numpy as np
import time
import functools

# matplotlib.get_backend()
# 'module://backend_interagg'


DEFAULT_FILE = '/home/qinglong/node3share/analytical_phantom_sinogram.h5'
file = tables.open_file(DEFAULT_FILE)

BATCH_SIZE = 32
DOWN_SAMPLING_RATIO = 2
SAMPLER_TARGET_SHAPE = [BATCH_SIZE, 16, 16, 1]


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

@clock
def show_aspr(imshow_index = 1):
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,4))
    ax1.set_title('align_cropped_low')
    ax1.imshow(align_cropped_low[imshow_index].reshape(SAMPLER_TARGET_SHAPE[1:3]))
    ax2.set_title('align_cropped_high')
    ax2.imshow(align_cropped_high[imshow_index]
               .reshape(list(np.multiply(SAMPLER_TARGET_SHAPE[1:3],DOWN_SAMPLING_RATIO))))
    plt.show()

@clock
def train(num_epoch=10):
    losses = {}
    counter = 0
    for _ in num_epoch:
        while True:
            try:
                _, temp_loss = sess.run([train_op, res['loss']])
                losses.append(temp_loss)
                counter += 1
                if counter % 100 == 0:
                    print(f'Loss after {counter} batches is {temp_loss}')
            except IndexError:
                break


d = DataGen(file, BATCH_SIZE)
ds = DownSampler(d.phantom, DOWN_SAMPLING_RATIO)
aspr = AlignSampler(ds(), d.phantom, SAMPLER_TARGET_SHAPE)
train_low, train_high = aspr()

superRe2x_ins = SuperResolution2x(
                                'sR',
                                inputs={'input': train_low, 'label': train_high},
                                nb_layers=2,
                                filters=5,
                                boundary_crop=[4, 4])
res = superRe2x_ins()
train_op = tf.train.AdamOptimizer(0.0001).minimize(res['loss'])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

train()

# show_aspr()




