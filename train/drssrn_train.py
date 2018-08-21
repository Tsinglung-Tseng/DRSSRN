from symbol.drssrn_utils import DataGen, DownSampler, AlignSampler
# from incident.
from dxl.learn.model import random_crop
import matplotlib.pyplot as plt
import tensorflow as tf
import tables

DEFAULT_FILE = '/home/qinglong/node3share/analytical_phantom_sinogram.h5'
file = tables.open_file(DEFAULT_FILE)
DOWN_SAMPLING_RATIO = 4
BATCH_SIZE = 32
CROP_TARGET_SHAPE = [BATCH_SIZE, 40, 40, 1]
SAMPLER_TARGET_SHAPE = [BATCH_SIZE, 8, 8, 1]

data_generator = DataGen(file, batch_size=BATCH_SIZE)

# shape=(?, 256, 256)
phantom, phantom_type, sinogram = data_generator.next_batch
phantom = tf.reshape(phantom, [BATCH_SIZE,256,256,1])

phantom_down_sampler = DownSampler(phantom, DOWN_SAMPLING_RATIO, data_generator.batch_size)
phantom_down_sample_x = phantom_down_sampler()

# shape=(32, 32, 32, 1)
# phantom_cropped = random_crop(phantom_down_sampler(), CROP_TARGET_SHAPE)

align_sampler = AlignSampler(phantom_down_sample_x, phantom, SAMPLER_TARGET_SHAPE)
sampled_low, sampled_high = align_sampler()

# img, label =

with tf.Session() as sess:
    img_down_sample_x2 = sess.run(phantom_down_sample_x)
    img_low_sampled = sess.run(sampled_low)
    img_high_sampled = sess.run(sampled_high)









# #
# img = img_down_sample_x2[14]
# # plt.imshow(img.reshape(phantom_down_sampler.output_size[1:3]))
# plt.imshow(img.reshape(CROP_TARGET_SIZE[1:3]))
# plt.colorbar()
# plt.show()
