from dxl.learn.core import Model
from dxl.learn.model.cnn import UpSampling2D, Conv2D
from typing import Dict
from dxl.learn.core import Model, Tensor
from dxl.learn.model.stack import Stack
from dxl.learn.model.residual import Residual
from dxl.learn.model.cnn import UpSampling2D, Conv2D
from dxl.learn.model.losses import mean_square_error, CombinedSupervisedLoss, poission_loss
from dxl.learn.model.crop import random_crop_offset, random_crop, boundary_crop, align_crop, shape_as_list
import tensorflow as tf
import numpy as np
import itertools
import tables
import math

from singledispatch import singledispatch
import matplotlib.pyplot as plt

DEFAULT_FILE = '/home/qinglong/node3share/analytical_phantom_sinogram.h5'
file = tables.open_file(DEFAULT_FILE)


class DataGen:
    """
    A data generator for phantom and sinogram dataset.
    g = DataGen(file)

    with tf.Session() as sess:
        sess.run(g.next_batch)
    """
    def __init__(self, _file, batch_size=32):
        self._file = _file
        self.batch_size = batch_size
        self.next_batch = (
            tf.data.Dataset
            .from_generator(self._gen,
                            (tf.float32, tf.int8, tf.float32),
                            (tf.TensorShape([256, 256]), tf.TensorShape([]), tf.TensorShape([320, 320])))
            .shuffle(buffer_size=1000)
            .batch(batch_size)
            .make_one_shot_iterator()
            .get_next()
        )

    def _gen(self):
        for i in itertools.count(780000):
            try:
                yield self._file.root.data[i][0], self._file.root.data[i][1], self._file.root.data[i][2]
            except IndexError:
                break

    @property
    def phantom(self):
        return tf.reshape(self.next_batch[0], [self.batch_size, 256, 256, 1])

    @property
    def sinogram(self):
        return tf.reshape(self.next_batch[2], [self.batch_size, 320, 320, 1])


class DownSampler:
    """
    d = DownSampler(input, down_sample_ratios, batch_size)
    with tf.Session() as sess:
        sess.run(d())
    """
    def __init__(self, input_, down_sample_ratios):
        self.input_ = input_
        self.down_sample_ratios = down_sample_ratios

    # @staticmethod
    # def shape_as_list(input):
    #     if isinstance(input, tf.Tensor):
    #         return list(input.shape.as_list())

    @property
    def batch_size(self):
        if self.input_dim == 4:
            return self.input_shape[0]
        if self.input_dim == 3:
            return 1
        else:
            raise ValueError(f'The shape of input {input_shape} is not accepted. Use 3D or 4D tensor.')

    @property
    def input_shape(self):
        return shape_as_list(self.input_)

    @property
    def input_dim(self):
        return len(shape_as_list(self.input_))

    @property
    def output_shape(self):
        return [self.input_shape[0]] + [self.input_shape[x]
                                        // self.down_sample_ratios
                                        for x in range(1, 3)] + [self.input_shape[3]]

    def __call__(self):
        return tf.image.resize_images(self.input_, tf.convert_to_tensor(self.output_shape[1:3], dtype=tf.int32))


class AlignSampler:
    def __init__(self, low, high, target_low_shape):
        self.low = low
        self.high = high
        self.target_low_shape = target_low_shape

    @property
    def low_shape(self):
        return shape_as_list(self.low)

    @property
    def high_shape(self):
        return shape_as_list(self.high)

    @property
    def target_high_shape(self):
        return list(np.multiply(self.target_low_shape, self.scale))

    @property
    def scale(self):
        return [x//y for x, y in zip(self.high_shape, self.low_shape)]

    @property
    def offsets(self):
        offset_low = random_crop_offset(self.low_shape, self.target_low_shape)
        offset_high = np.multiply(self.scale, offset_low)
        return list(offset_low), list(offset_high)

    def __call__(self):
        offset_low, offset_high = self.offsets
        return (tf.slice(self.low, offset_low, self.target_low_shape),
                tf.slice(self.high, offset_high, self.target_high_shape))


def show():
    pass


def psnr(inference, label, pix_max=255.0, idx=0):
    mse = np.mean((rescale(inference[idx]) - rescale(label[idx])) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(pix_max / math.sqrt(mse))

def rescale(inputs, bin_size=255):
    window = inputs.max() - inputs.min()
    scale_rate = bin_size / window
    return inputs*scale_rate - inputs.min()*scale_rate








class SRKeys:
    REPRESENTS = 'reps'
    RESIDUAL = 'resi'
    ALIGNED_LABEL = 'aligned_label'
    INTERP = 'interp'
    POI_LOSS = 'poi_loss'
    MSE_LOSS = 'mse_loss'


class SuperResolution2x(Model):
    """ SuperResolution2x Block
    Arguments:
        name: Path := dxl.fs
            A unique block name
        inputs: Dict[str, Tensor/tf.Tensor] input.
        nb_layers: integer.
        filters: Integer, the dimensionality of the output space.
        boundary_crop: Tuple/List of 2 integers.
        graph: kernel
            One of StackedConv2D/StackedResidualConv/StackedResidualIncept
    """

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            INFERENCE = 'inference'
            LABEL = 'label'
            LOSS = 'loss'

        class CONFIG:
            NB_LAYERS = 'nb_layers'
            FILTERS = 'filters'
            BOUNDARY_CROP = 'boundary_crop'

        class GRAPHS:
            SHORT_CUT = 'buildingblock'

    def __init__(self,
                 info,
                 inputs,
                 nb_layers=None,
                 filters=None,
                 boundary_crop=None,
                 graph=None):
        super().__init__(
            info,
            tensors=inputs,
            graphs={self.KEYS.GRAPHS.SHORT_CUT: graph},
            config={
                self.KEYS.CONFIG.NB_LAYERS: nb_layers,
                self.KEYS.CONFIG.FILTERS: filters,
                self.KEYS.CONFIG.BOUNDARY_CROP: boundary_crop,
            })

    @classmethod
    def _default_config(cls):
        return {
            cls.KEYS.CONFIG.NB_LAYERS: 2,
            cls.KEYS.CONFIG.FILTERS: 5,
            cls.KEYS.CONFIG.BOUNDARY_CROP: (4, 4)
        }

    def _short_cut(self, name):
        conv2d_ins = Conv2D(
            info="conv2d",
            filters=self.config(self.KEYS.CONFIG.FILTERS),
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='basic')
        return Stack(
            info=self.info.child_scope(name),
            models=conv2d_ins,
            nb_layers=2)

    def kernel(self, inputs):
        with tf.variable_scope('input'):
            u = UpSampling2D(
                inputs=inputs[self.KEYS.TENSOR.INPUT], size=(2, 2))()
            if SRKeys.REPRESENTS in inputs:
                r = UpSampling2D(
                    inputs=inputs[SRKeys.REPRESENTS], size=(2, 2))()
                r = align_crop(r, u)
                r = tf.concat([r, u], axis=3)
            else:
                r = tf.layers.conv2d(
                    inputs=u,
                    filters=self.config(self.KEYS.CONFIG.FILTERS),
                    kernel_size=5,
                    name='stem0')

        key = self.KEYS.GRAPHS.SHORT_CUT
        x = self.get_or_create_graph(key, self._short_cut(key))(r)
        with tf.variable_scope('inference'):
            res = tf.layers.conv2d(
                inputs=x,
                filters=1,
                kernel_size=3,
                padding='same',
                name='stem1',
            )
            res = boundary_crop(res,
                                self.config(self.KEYS.CONFIG.BOUNDARY_CROP))
            u_c = align_crop(u, res)
            y = res + u_c

        result = {
            self.KEYS.TENSOR.INFERENCE: y,
            SRKeys.REPRESENTS: x,
            SRKeys.RESIDUAL: res,
            SRKeys.INTERP: u_c
        }
        if self.KEYS.TENSOR.LABEL in inputs:
            with tf.name_scope('loss'):
                aligned_label = align_crop(inputs[self.KEYS.TENSOR.LABEL], y)
                l = mean_square_error(aligned_label, y)
            result.update({
                self.KEYS.TENSOR.LOSS: l,
                SRKeys.ALIGNED_LABEL: aligned_label
            })

        return result

def get_data():
    DEFAULT_FILE = '/home/qinglong/node3share/analytical_phantom_sinogram.h5'
    file = tables.open_file(DEFAULT_FILE)

    class FLAGS:
        class TRAIN:
            BATCH_SIZE = 32
            DOWN_SAMPLING_RATIO = 2
            SAMPLER_TARGET_SHAPE = [BATCH_SIZE, 64, 64, 1]

        class SUMMARY:
            SUMMARY_DIR = '/home/qinglong/node3share/remote_drssrn/tensorboard_log/drssrn_3_epoch'

    d = DataGen(file, FLAGS.TRAIN.BATCH_SIZE)
    ds = DownSampler(d.phantom, FLAGS.TRAIN.DOWN_SAMPLING_RATIO)
    aspr = AlignSampler(ds(), d.phantom, FLAGS.TRAIN.SAMPLER_TARGET_SHAPE)
    return aspr()




class DataIte:
    """
    i = DataIte
    """
    def __init__(self):
        ds = tf.data.Dataset.range(25,300)
        ds = ds.shuffle(buffer_size=10)
        ds = ds.batch(32) #batch_size=32

        a_iterator = tf.data.Iterator.from_structure(tf.int64, tf.TensorShape([]))
        prediction = a_iterator.get_next()

        ds_iterator = a_iterator.make_initializer(ds)

        with tf.Session() as sess:
            sess.run(ds_iterator)
            while True:
                try:
                    print(sess.run(prediction))
                except tf.errors.OutOfRangeError:
                    break
