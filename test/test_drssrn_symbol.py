from src.drssrn_symbol import residual_block, inference_block
import tensorflow as tf
import numpy as np
import keras.backend as K
import unittest

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

tf.reset_default_graph()

class Test_drssrn_symbol(unittest.TestCase):

    def test_residual_block(self):
        with tf.Session(config=config) as test:
            np.random.seed(1)

            A_prev = tf.placeholder("float", [3, 256, 256, 32])
            X = np.random.randn(3, 256, 256, 32)

            A = residual_block(A_prev, scope = 'a')

            test.run(tf.global_variables_initializer())
            X_out = test.run(A, feed_dict={A_prev: X, K.learning_phase(): 0})
            self.assertEqual(np.shape(X_out), (3,256,256,32))

    def test_residual_block_output_type(self):
        with tf.Session(config=config) as test:
            np.random.seed(1)
            A_prev = tf.placeholder("float", [3, 256, 256, 32])

            A = residual_block(A_prev, scope = 'a')
            # TODO change to assertTure(isinstance(..., ...))
            self.assertEqual(str(type(A)), "<class 'tensorflow.python.framework.ops.Tensor'>")

    def test_inference_block(self):
        with tf.Session(config=config) as test:
            np.random.seed(1)
            inference_in = tf.placeholder('float', [3, 256, 256, 32])
            X = np.random.randn(3, 256, 256, 32)

            inference_out = inference_block(inference_in)
            test.run(tf.initialize_all_variables())
            X_out = test.run(inference_out, feed_dict={inference_in: X})
            self.assertEqual(np.shape(X_out), (3, 256, 256, 32))


if __name__ == "__main__":
    unittest.main()