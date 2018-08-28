from symbol.drssrn_utils import DataGen, DownSampler, AlignSampler, psnr, rescale
# from incident.
from dxl.learn.model.super_resolution import SuperResolution2x, SuperResolutionBlock
from dxl.learn.model.crop import random_crop_offset, random_crop, boundary_crop, align_crop, shape_as_list
from dxl.learn.model import random_crop
import tensorflow as tf
import tables
import numpy as np
import time
import functools
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from symbol._old_drssrn_symbol import inference_block

DEFAULT_FILE = '/home/qinglong/node3share/analytical_phantom_sinogram.h5'
file = tables.open_file(DEFAULT_FILE)


class FLAGS:
    class TRAIN:
        BATCH_SIZE = 32
        DOWN_SAMPLING_RATIO = 2
        SAMPLER_TARGET_SHAPE = [BATCH_SIZE, 64, 64, 1]

    class SUMMARY:
        SUMMARY_DIR = '/home/qinglong/node3share/remote_drssrn/tensorboard_log/5x_64_0.00001_with_stand&rescale'


def show_subplot(interp, inference, label, psnr, counter, ind=0):
    plt.figure(1)
    plt.suptitle(f"PSNR after {counter} batches: " + str(psnr))

    plt.subplot(131)
    plt.title('interp')
    plt.imshow(interp[ind].reshape(interp.shape[1:3]))

    plt.subplot(132)
    plt.title('inference')
    plt.imshow(inference[ind].reshape(inference.shape[1:3]))

    plt.subplot(133)
    plt.title('label')
    plt.imshow(label[ind].reshape(label.shape[1:3]))
    plt.show()


def residuals(x):
    for i in range(20):
        with tf.variable_scope(f'layer_{i}'):
            h = tf.layers.conv2d(x, 64, 3, activation=tf.nn.elu, padding='same')
            x = x + 0.3 * h
    return x


def inference_block(x):
    for i in range(5):
        with tf.variable_scope(f'infer_layer_{i}'):
            x_a = x
            x_b = tf.nn.relu(x)

            x_b1 = tf.layers.conv2d(x_b, 64, 1, activation=None, padding='same')

            x_b2 = tf.layers.conv2d(x_b, 64, 1, activation=tf.nn.elu, padding='same')
            x_b2 = tf.layers.conv2d(x_b2, 64, 3, activation=None, padding='same')

            x_b3 = tf.layers.conv2d(x_b, 64, 1, activation=tf.nn.elu, padding='same')
            x_b3 = tf.layers.conv2d(x_b3, 64, 3, activation=tf.nn.elu, padding='same')
            x_b3 = tf.layers.conv2d(x_b3, 64, 3, activation=None, padding='same')

            x_bc = tf.concat([x_b1, x_b2, x_b3], axis=3)
            x_bc = tf.nn.relu(x_bc)
            x_bc = tf.layers.conv2d(x_bc, 64, 3, activation=None, padding='same')

            res_block = x_a + 0.3 * x_bc
            x = x + res_block

    return x


def rs(images):
    return np.array([rescale(img) for img in images])


d = DataGen(file, FLAGS.TRAIN.BATCH_SIZE)
ds = DownSampler(d.phantom, FLAGS.TRAIN.DOWN_SAMPLING_RATIO)
aspr = AlignSampler(ds(), d.phantom, FLAGS.TRAIN.SAMPLER_TARGET_SHAPE)
train_low, train_high = aspr()

train_high_shape = shape_as_list(train_high)
train_low_shape = shape_as_list(train_low)

train_high = tf.map_fn(tf.image.per_image_standardization, train_high)
train_low = tf.map_fn(tf.image.per_image_standardization, train_low)

train_high = tf.reshape(tf.py_func(rs, [train_high], tf.float32), train_high_shape)
train_low = tf.reshape(tf.py_func(rs, [train_low], tf.float32), train_low_shape)


with tf.device('/device:GPU:1'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True

    superRe2x_ins = SuperResolution2x('sR',
                                      inputs={'input': train_low, 'label': train_high},
                                      nb_layers=5,
                                      filters=64,
                                      boundary_crop=[4, 4],
                                      graph=inference_block)
    res = superRe2x_ins()
# ****************** with ends here ******************

tf_psnr = tf.image.psnr(res['inference'][0], res['aligned_label'][0], max_val=255)
train_op = tf.train.AdamOptimizer(0.00001).minimize(res['loss'])

tf.summary.scalar("loss", res['loss'])
tf.summary.scalar("psnr", tf_psnr)
merged_summary = tf.summary.merge_all()

sess = tf.Session(config=config)
writer = tf.summary.FileWriter(FLAGS.SUMMARY.SUMMARY_DIR, sess.graph)

sess.run(tf.global_variables_initializer())

counter = 0
print("Training...")
while True:
    try:
        (_,
         summary,
         loss_temp,
         inference,
         aligned_label,
         reps,
         resi,
         interp) = sess.run([train_op,
                             merged_summary,
                             res['loss'],
                             res['inference'],
                             res['aligned_label'],
                             res['reps'],
                             res['resi'],
                             res['interp']])
        writer.add_summary(summary, counter)

        counter += 1
        if counter % 1000 == 0:
            print(f'Loss after {counter} batch is {loss_temp}')
            temp_psnr = psnr(inference, aligned_label)
            show_subplot(interp, inference, aligned_label, psnr=temp_psnr, counter=counter)

    except tf.errors.OutOfRangeError:
        print('Done')
        break





# def summary():










