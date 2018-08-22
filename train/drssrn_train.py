from symbol.drssrn_utils import DataGen, DownSampler, AlignSampler
# from incident.
from dxl.learn.model import random_crop
import matplotlib.pyplot as plt
import tensorflow as tf
import tables
import numpy as np

# matplotlib.get_backend()
# 'module://backend_interagg'


DEFAULT_FILE = '/home/qinglong/node3share/analytical_phantom_sinogram.h5'
file = tables.open_file(DEFAULT_FILE)

BATCH_SIZE = 32
DOWN_SAMPLING_RATIO = 4
SAMPLER_TARGET_SHAPE = [BATCH_SIZE, 16, 16, 1]

def show_aspr(imshow_index = 1):
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,4))
    ax1.set_title('align_cropped_low')
    ax1.imshow(align_cropped_low[imshow_index].reshape(SAMPLER_TARGET_SHAPE[1:3]))
    ax2.set_title('align_cropped_high')
    ax2.imshow(align_cropped_high[imshow_index]
               .reshape(list(np.multiply(SAMPLER_TARGET_SHAPE[1:3],DOWN_SAMPLING_RATIO))))
    plt.show()


d = DataGen(file, BATCH_SIZE)
ds = DownSampler(d.phantom, DOWN_SAMPLING_RATIO)
aspr = AlignSampler(ds(), d.phantom, SAMPLER_TARGET_SHAPE)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

with tf.Session(config=config) as sess:
    align_cropped_low, align_cropped_high = sess.run(aspr())


show_aspr()




