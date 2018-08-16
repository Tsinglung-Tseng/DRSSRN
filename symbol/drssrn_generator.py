import tensorflow as tf
import itertools
import tables

DEFAULT_FILE = '/home/qinglong/node3share/analytical_phantom_sinogram.h5'
file = tables.open_file(DEFAULT_FILE)


class DataGen:
    """
    A data generator for phantom and sinogram dataset.
    g = DataGen(file)

    with tf.Session() as sess:
        sess.run(g.next_)
    """
    def __init__(self, file, batch_size=32):
        self.file = file
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
            yield self.file.root.data[i][0], self.file.root.data[i][1], self.file.root.data[i][2]


class DataIte:
    def __init__(self):
        ds = tf.data.Dataset.range(25,300)
        ds = ds.shuffle(buffer_size=10)
        ds = ds.batch(BATCH_SIZE)


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


