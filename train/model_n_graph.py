import tensorflow as tf
import numpy as np
from typing import Dict
from dxl.learn.core import Model, Tensor
from dxl.learn.model.stack import Stack
from dxl.learn.model.residual import Residual
from dxl.learn.model.crop import boundary_crop, align_crop, shape_as_list
from dxl.learn.model.cnn import UpSampling2D, Conv2D
from dxl.learn.model.losses import mean_square_error, CombinedSupervisedLoss, poission_loss


class SRCNN(Model):

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            INFERENCE = 'inference'
            LABEL = 'label'
            LOSS = 'loss'

        class CONFIG:
            NB_LAYERS = 'nb_layer'
            FILTERS = 'filters'
            BOUNDARY_CROP = 'boundary_crop'

        class GRAPHS:
            SHORT_CUT = 'buildingblock'




dummy_input = tf.constant(np.arange(6144), shape=[32,8,8,3], dtype=tf.float32)

cv2 = Conv2D(inputs=dummy_input, filters=3, kernel_size=(3,3))

# cv2 = tf.layers.conv2d(inputs=dummy_input, filters=3, kernel_size=(3,3))

res = cv2()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(res)