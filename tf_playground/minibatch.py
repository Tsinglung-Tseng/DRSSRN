import tensorflow as tf
import itertools
import tables
import matplotlib.pyplot as plt

DEFAULT_FILE = '/home/qinglong/node3share/analytical_phantom_sinogram.h5'

# TODO replace with parser
# iterator (one-shot)
# def sino_gen():
#     for i in itertools.count(1):
#         yield tables.open_file(DEFAULT_FILE).root.data[i][2]
#
# def img_gen():
#     for i in itertools.count(1):
#         yield tables.open_file(DEFAULT_FILE).root.data[i][0]
#
# next_sino = (tf.data.Dataset
#              .from_generator(sino_gen, tf.float32, tf.TensorShape([320,320]))
#              .make_one_shot_iterator()
#              .get_next())
#
# next_img = (tf.data.Dataset
#              .from_generator(img_gen, tf.float32, tf.TensorShape([256,256]))
#              .make_one_shot_iterator()
#              .get_next())
#
#
# sess = tf.Session()
# v = sess.run(next_img)
#
#
# # batch
# batched_sino = (tf.data.Dataset
#              .from_generator(sino_gen, tf.float32, tf.TensorShape([320,320]))
#              .batch(32))
#
# iterator = batched_sino.make_one_shot_iterator()
# batched_sino_iter = iterator.get_next()
#
# bc = sess.run(batched_sino_iter)


class gen:
    def __init__(self, DEFAULT_FILE, batch_size=32):
        self.DEFAULT_FILE = DEFAULT_FILE
        self.next_sino = (tf.data.Dataset
                         .from_generator(self._sino_gen, tf.float32, tf.TensorShape([320,320]))
                         .make_one_shot_iterator()
                         .get_next())

        self.next_img = (tf.data.Dataset
                        .from_generator(self._img_gen, tf.float32, tf.TensorShape([256,256]))
                        .make_one_shot_iterator()
                        .get_next())

    def _sino_gen():
        for i in itertools.count(1):
            # TODO uncoupling path
            yield tables.open_file(DEFAULT_FILE).root.data[i][2]

    def _img_gen():
        for i in itertools.count(1):
            yield tables.open_file(DEFAULT_FILE).root.data[i][0]

class batched_gen:

    def parse(serialized):

        features = \
            {
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            }


        parsed_example = tf.parse_single_example(serialized=serialized,
                                                 features=features)

        # Get the image as raw bytes.
        image_raw = parsed_example['image']

        # Decode the raw bytes so it becomes a tensor with type.
        image = tf.decode_raw(image_raw, tf.uint8)

        # The type is now uint8 but we need it to be float.
        image = tf.cast(image, tf.float32)

        # Get the label associated with the image.
        label = parsed_example['label']

        # The image and label are now correct TensorFlow types.
        return image, label