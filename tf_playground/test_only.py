import tensorflow as tf
import tables
import numpy as np
#from dxl.learn.model import random_crop

DEFAULT_FILE = '/home/qinglong/node3share/analytical_phantom_sinogram.h5'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

f = tables.open_file(DEFAULT_FILE)

sino_ph = tf.placeholder(tf.float32, shape=[None,320,320])
sino_raw,*_ = f.root.data

mb = tf.train.batch(
    tensors=sino_raw,
    batch_size=64,
    num_threads=1,
    capacity=32,
    enqueue_many=False,
    shapes=None,
    dynamic_pad=False,
    allow_smaller_final_batch=True,
    shared_name=None,
    name=None
)



# down_sampled = tf.image.central_crop(
#     image=mb,
#     central_fraction
# )

sess = tf.Session(config=config)
sess.run(tf.initialize_all_variables())
sess.run(mb)