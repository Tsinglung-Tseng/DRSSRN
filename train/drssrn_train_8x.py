from symbol.drssrn_utils import DataGen, DownSampler, AlignSampler, psnr, rescale_batch
from dxl.learn.model.super_resolution import SuperResolution2x
from dxl.learn.model.cnn.blocksv2 import Conv2D, Inception, Residual
from dxl.learn.model.cnn import DownSampling2D
from dxl.learn.model.base import Stack, as_model, Model
from doufo import List, identity
from dxl.learn.model.crop import shape_as_list
import matplotlib.pyplot as plt
import tensorflow as tf
import tables


DEFAULT_FILE = '/home/qinglong/node3share/analytical_phantom_sinogram.h5'
file = tables.open_file(DEFAULT_FILE)


class FLAGS:
    class TRAIN:
        BATCH_SIZE = 32
        DOWN_SAMPLING_RATIO = 8
        SAMPLER_TARGET_SHAPE = [BATCH_SIZE, 16, 16, 1]

    class SUMMARY:
        SUMMARY_DIR = '/home/qinglong/node3share/remote_drssrn/tensorboard_log/8xDown_3'


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


def incept_path(ipath, filters):
    result = List([])
    result.append(as_model(tf.nn.elu))
    result.append(Conv2D(f'conv_in_{ipath}', filters, 1))
    for i in range(ipath):
        result.append(as_model(tf.nn.elu))
        result.append(Conv2D(f"conv_{ipath}_{i}", filters, 3))
    return Stack(result)


class Merge(Model):
    def __init__(self, filters, name='merger'):
        super().__init__(name)
        self.filters = filters
        self.model = Conv2D('conv_merge', filters, 3)

    @property
    def parameters(self):
        return self.model.parameters

    def kernel(self, xs):
        x = tf.concat(xs, axis=3)
        return self.model(x)


def SRx2(sr_block, num_layer=5):
    SRx2 = list([])
    for i in range(num_layer):
        SRx2.append(Residual(f'residual_{i}_{sr_block}',
                    Inception(f'incept_{i}_{sr_block}', identity,
                              [incept_path(i, 64) for i in range(3)],
                              Merge(64)),
                    ratio=0.3))
    return Stack(SRx2)


d = DataGen(file, FLAGS.TRAIN.BATCH_SIZE)
ds = DownSampler(d.phantom, FLAGS.TRAIN.DOWN_SAMPLING_RATIO)
aspr = AlignSampler(ds(), d.phantom, FLAGS.TRAIN.SAMPLER_TARGET_SHAPE)
train_low, train_high = aspr()

# reminder -- converting images from scale 65535 to 255
train_high_shape = shape_as_list(train_high)
train_low_shape = shape_as_list(train_low)

train_high = tf.map_fn(tf.image.per_image_standardization, train_high)
train_low = tf.map_fn(tf.image.per_image_standardization, train_low)

train_high = tf.reshape(tf.py_func(rescale_batch, [train_high], tf.float32), train_high_shape)
train_low = tf.reshape(tf.py_func(rescale_batch, [train_low], tf.float32), train_low_shape)

downsampling2d_x2 = DownSampling2D("res_0_downsampling",
                                   inputs=train_high,
                                   size=[2*i for i in FLAGS.TRAIN.SAMPLER_TARGET_SHAPE[1:3]],
                                   is_scale=False,
                                   method=2)
train_high_x2 = downsampling2d_x2()

downsampling2d_x4 = DownSampling2D("res_1_downsampling",
                                   inputs=train_high,
                                   size=[4*i for i in FLAGS.TRAIN.SAMPLER_TARGET_SHAPE[1:3]],
                                   is_scale=False,
                                   method=2)
train_high_x4 = downsampling2d_x4()


with tf.device('/device:GPU:1'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True

    superRe2x_ins_0 = SuperResolution2x('sR_0',
                                        inputs={'input': train_low, 'label': train_high_x2},
                                        nb_layers=5,
                                        filters=64,
                                        boundary_crop=[4, 4],
                                        graph=SRx2(sr_block=0))
    res_0 = superRe2x_ins_0()

    superRe2x_ins_1 = SuperResolution2x('sR_1',
                                        inputs={'input': res_0['inference'],
                                                'label': train_high_x4},
                                        nb_layers=5,
                                        filters=64,
                                        boundary_crop=[4, 4],
                                        graph=SRx2(sr_block=1))
    res_1 = superRe2x_ins_1()

    superRe2x_ins_2 = SuperResolution2x('sR_2',
                                        inputs={'input': res_1['inference'],
                                                'label': train_high},
                                        nb_layers=5,
                                        filters=64,
                                        boundary_crop=[4, 4],
                                        graph=SRx2(sr_block=2))
    res_2 = superRe2x_ins_2()

    loss = 0.1*res_0['loss'] + 0.3*res_1['loss'] + res_2['loss']


tf_psnr = tf.image.psnr(res_1['inference'][0], res_1['aligned_label'][0], max_val=255)
train_op = tf.train.AdamOptimizer(0.00001).minimize(loss)

tf.summary.scalar("loss", res_1['loss'])
tf.summary.scalar("psnr", tf_psnr)
merged_summary = tf.summary.merge_all()

sess = tf.Session(config=config)
writer = tf.summary.FileWriter(FLAGS.SUMMARY.SUMMARY_DIR, sess.graph)

sess.run(tf.global_variables_initializer())


counter = 0
print("Training8x...")
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
                             res_2['loss'],
                             res_2['inference'],
                             res_2['aligned_label'],
                             res_2['reps'],
                             res_2['resi'],
                             res_2['interp']])
        writer.add_summary(summary, counter)

        counter += 1
        if counter % 100 == 0:
            print(f'Loss after {counter} batch is {loss_temp}')
            temp_psnr = psnr(inference, aligned_label)
            show_subplot(interp, inference, aligned_label, psnr=temp_psnr, counter=counter)

    except tf.errors.OutOfRangeError:
        print('Done')
        break