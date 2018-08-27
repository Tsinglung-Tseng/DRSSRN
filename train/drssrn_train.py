from symbol.drssrn_utils import DataGen, DownSampler, AlignSampler, psnr
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
from sklearn.preprocessing import MinMaxScaler

# matplotlib.get_backend()
# 'module://backend_interagg'


DEFAULT_FILE = '/home/qinglong/node3share/analytical_phantom_sinogram.h5'
file = tables.open_file(DEFAULT_FILE)

class FLAGS:
    class TRAIN:
        BATCH_SIZE = 32
        DOWN_SAMPLING_RATIO = 2
        SAMPLER_TARGET_SHAPE = [BATCH_SIZE, 64, 64, 1]

    class SUMMARY:
        SUMMARY_DIR = '/home/qinglong/node3share/remote_drssrn/tensorboard_log/drssrn_3_epoch'


def show_subplot(img, label, psnr,ind=0):
    plt.figure(1)
    plt.suptitle("PSNR: " + str(psnr))
    plt.subplot(121)
    plt.imshow(img[ind].reshape(img.shape[1:3]))

    plt.subplot(122)
    plt.imshow(label[ind].reshape(label.shape[1:3]))
    plt.show()


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

tf.summary.scalar("loss_0", res['loss'])
merged_summary = tf.summary.merge_all()

sess = tf.Session(config=config)
writer = tf.summary.FileWriter(FLAGS.SUMMARY.SUMMARY_DIR, sess.graph)

sess.run(tf.global_variables_initializer())

counter = 0
while True:
    try:
        (_,
         summary,
         loss_temp,
         inference,
         aligned_label) = sess.run([train_op,
                                    merged_summary,
                                    res['loss'],
                                    res['inference'],
                                    res['aligned_label']])
        writer.add_summary(summary, counter)

        temp_psnr = psnr(inference, aligned_label)
        counter += 1
        if counter % 100 == 0:
            print(f'Loss after {counter} batch is {loss_temp}')

    except tf.errors.OutOfRangeError:
        print('Done')
        break

show_subplot(inference, aligned_label,temp_psnr)



# def summary():










