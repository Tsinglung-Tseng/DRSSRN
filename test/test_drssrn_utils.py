from symbol.drssrn_utils import DataGen
import tensorflow as tf
import tables
import unittest
import numpy as np

DEFAULT_FILE = '/home/qinglong/node3share/analytical_phantom_sinogram.h5'
file = tables.open_file(DEFAULT_FILE)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

class TestDrssrnUtils(unittest.TestCase):
    def test_DataGen(self):
        test_gen = DataGen(file)
        with tf.Session(config=config) as test:
            phantom = test.run(test_gen.phantom)
            batch_size = test_gen.batch_size
            self.assertEqual(phantom.shape, (batch_size, 256, 256, 1))

