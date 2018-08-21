from symbol.drssrn_utils import DataGen, DownSampler
from dxl.learn.model import random_crop
import matplotlib.pyplot as plt
import tensorflow as tf
import tables

DEFAULT_FILE = '/home/qinglong/node3share/analytical_phantom_sinogram.h5'
file = tables.open_file(DEFAULT_FILE)
DOWN_SAMPLING_RATIO = 8
BATCH_SIZE = 32
CROP_TARGET_SIZE = [BATCH_SIZE,8,8,1]

data_generator = DataGen(file, batch_size=BATCH_SIZE)
phantom, phantom_type, sinogram = data_generator.next_batch



phantom_down_sampler = DownSampler(phantom, DOWN_SAMPLING_RATIO, data_generator.batch_size)


phantom_cropped = random_crop(phantom_down_sampler(), CROP_TARGET_SIZE)
# img, label =

with tf.Session() as sess:
    img_down_sample_x2 = sess.run(phantom_cropped)











img = img_down_sample_x2[14]
# plt.imshow(img.reshape(phantom_down_sampler.output_size[1:3]))
plt.imshow(img.reshape(CROP_TARGET_SIZE[1:3]))
plt.colorbar()
plt.show()
