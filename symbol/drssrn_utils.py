from dxl.learn.model.crop import random_crop_offset, random_crop, boundary_crop, align_crop, shape_as_list
import tensorflow as tf
import itertools
import tables
import numpy as np
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
        for i in itertools.count(1):
            yield self._file.root.data[i][0], self._file.root.data[i][1], self._file.root.data[i][2]

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
