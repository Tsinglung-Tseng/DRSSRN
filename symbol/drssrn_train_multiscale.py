from symbol.drssrn_utils import DataGen, DownSampler, AlignSamplernvid, psnr
from train.drssrn_train import show_subplot, residuals, inference_block,
import tensorflow as tf
import tables


DEFAULT_FILE = '/home/qinglong/node3share/analytical_phantom_sinogram.h5'
file = tables.open_file(DEFAULT_FILE)


class FLAGS:
    class TRAIN:
        BATCH_SIZE = 32
        DOWN_SAMPLING_RATIO = 4
        SAMPLER_TARGET_SHAPE = [BATCH_SIZE, 64, 64, 1]

    class SUMMARY:
        SUMMARY_DIR = '/home/qinglong/node3share/remote_drssrn/tensorboard_log/drssrn_inferenceblock'


d = DataGen(file, FLAGS.TRAIN.BATCH_SIZE)
ds = DownSampler(d.phantom, FLAGS.TRAIN.DOWN_SAMPLING_RATIO)
aspr = AlignSampler(ds(), d.phantom, FLAGS.TRAIN.SAMPLER_TARGET_SHAPE)
train_low, train_high = aspr()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

with tf.device('/device:GPU:1'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True

    superRe2x_ins = SuperResolution2x('sR',
                                      inputs={'input': train_low, 'label': train_high},
                                      nb_layers=5,
                                      filters=32,
                                      boundary_crop=[4, 4],
                                      graph=inference_block(num_layers=10))
    res = superRe2x_ins()

tf_psnr = tf.image.psnr(res['inference'][0], res['aligned_label'][0], max_val=1.0)
train_op = tf.train.AdamOptimizer(0.00001).minimize(res['loss'])

tf.summary.scalar("loss", res['loss'])
tf.summary.scalar("psnr", tf_psnr)

