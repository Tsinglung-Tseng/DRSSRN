from symbol.drssrn_utils import DataGen, DownSampler, AlignSampler, psnr
from train.drssrn_train import show_subplot, residuals, inference_block, rs
from dxl.learn.model.super_resolution import SuperResolution2x
from dxl.learn.model.crop import shape_as_list
import tensorflow as tf
import tables


DEFAULT_FILE = '/home/qinglong/node3share/analytical_phantom_sinogram.h5'
file = tables.open_file(DEFAULT_FILE)


class FLAGS:
    class TRAIN:
        BATCH_SIZE = 32
        DOWN_SAMPLING_RATIO = 4
        SAMPLER_TARGET_SHAPE = [BATCH_SIZE, 32, 32, 1]

    class SUMMARY:
        SUMMARY_DIR = '/home/qinglong/node3share/remote_drssrn/tensorboard_log/drssrn_inferenceblock'


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
print("Training4x...")
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

