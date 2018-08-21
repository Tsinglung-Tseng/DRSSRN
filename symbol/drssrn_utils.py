# from dxl.learn.model import shape_as_list, random_crop_offset, random_crop,
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
        self.next_one = (
            tf.data.Dataset
            .from_generator(self._gen,
                            (tf.float32, tf.int8, tf.float32),
                            (tf.TensorShape([256, 256]), tf.TensorShape([]), tf.TensorShape([320, 320])))
            .make_one_shot_iterator()
            .get_next()
        )

        self.next_batch = (
            tf.data.Dataset
            .from_generator(self._gen,
                            (tf.float32, tf.int8, tf.float32),
                            (tf.TensorShape([256, 256]), tf.TensorShape([]), tf.TensorShape([320, 320])))
            .batch(batch_size)
            .make_one_shot_iterator()
            .get_next()
        )

    def _gen(self):
        for i in itertools.count(1):
            # TODO uncoupling path
            yield self._file.root.data[i][0], self._file.root.data[i][1], self._file.root.data[i][2]


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


class DownSampler:
    """
    d = DownSampler(input, down_sample_ratios, batch_size)
    with tf.Session() as sess:
        sess.run(d())
    """
    def __init__(self, input, down_sample_ratios, batch_size):
        self.input = input
        self.down_sample_ratios = down_sample_ratios
        self.batch_size = batch_size
        self.input_dim = len(self.shape_as_list(input))
        self.output_size = self._output_size()

    @staticmethod
    def shape_as_list(input):
        if isinstance(input, tf.Tensor):
            return list(input.shape.as_list())

    def _output_size(self):
        if (self.input_dim != 2) and (self.input_dim != 3):
            raise ValueError("Unsupported tensor dimension: {}, only 2D or 3D tensors are accepted.".format(self.input_dim))
        else:
            if self.input_dim == 2:
                return self._get_down_sample_img_size(self.shape_as_list(self.input))
            else:
                tensor_dim = self.shape_as_list(self.input)
                img_dim = self._get_down_sample_img_size(tensor_dim[1:])
                img_dim.insert(0, self.batch_size)
                img_dim.append(1)
                return img_dim

    def _get_down_sample_img_size(self, origin_dim):
        return list(np.ceil(np.divide(origin_dim, self.down_sample_ratios)).astype(np.int32))

    def shape_plus(self):
        if (self.input_dim != 2) and (self.input_dim != 3):
            raise ValueError("Unsupported tensor dimension: {}, only 2D or 3D tensors are accepted.".format(self.input_dim))
        else:
            if self.input_dim == 2:
                shape_plus = self.shape_as_list(self.input)
                shape_plus.append(1)
                return shape_plus
            else:
                shape_plus = self.shape_as_list(self.input)
                shape_plus[0] = self.batch_size
                shape_plus.append(1)
                return shape_plus

    def __call__(self):
        shape_plus = self.shape_plus()
        input_reshaped = tf.reshape(self.input, shape_plus)
        if self.input_dim == 2:
            return tf.image.resize_images(input_reshaped, tf.convert_to_tensor(self.output_size, dtype=tf.int32))
        else:
            return tf.image.resize_images(input_reshaped, tf.convert_to_tensor(self.output_size[1:3], dtype=tf.int32))

# class AlignedCroper:



# if __name__ == '__main__':
#     g = DataGen(file)
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         a,b,c = g.next_batch
#
#         d = DownSampler(a, 8, g.batch_size)
#
#     with tf.Session() as sess:
#         img_down_sample_x2 = sess.run(d())
#
#     # plt.imshow(img_down_sample_x2.reshape(d.output_size))
#     # plt.colorbar()
#     # plt.show()
#
#     # img = img_down_sample_x2[1]
#     # plt.imshow(img.reshape(d.output_size[1:3]))
#     # plt.colorbar()
#     # plt.show()

# class gen:
#     """
#     g = gen(file)
#
#     with tf.Session() as sess:
#         sess.run(g.next_sino_batch)
#     """
#     def __init__(self, file, batch_size=32):
#         self.file = file
#         self.next_sino = (tf.data.Dataset
#                          .from_generator(self._sino_gen, tf.float32, tf.TensorShape([320,320]))
#                          .make_one_shot_iterator()
#                          .get_next())
#
#         self.next_img = (tf.data.Dataset
#                         .from_generator(self._img_gen, tf.float32, tf.TensorShape([256,256]))
#                         .make_one_shot_iterator()
#                         .get_next())
#
#         self.next_sino_batch = (tf.data.Dataset
#                               .from_generator(self._sino_gen, tf.float32, tf.TensorShape([320,320]))
#                               .batch(batch_size)
#                               .make_one_shot_iterator()
#                               .get_next())
#
#         self.next_img_batch = (tf.data.Dataset
#                              .from_generator(self._img_gen, tf.float32, tf.TensorShape([256,256]))
#                              .batch(batch_size)
#                              .make_one_shot_iterator()
#                              .get_next())
#
#     def _sino_gen(self):
#         for i in itertools.count(1):
#
#             yield self.file.root.data[i][2]
#
#     def _img_gen(self):
#         for i in itertools.count(1):
#             yield self.file.root.data[i][0]