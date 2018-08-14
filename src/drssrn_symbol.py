from keras.layers import Concatenate, Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform
import tensorflow as tf
import keras.backend as K

K.set_image_data_format('channels_last')
K.set_learning_phase(1)

def inference_block(X, n_res = 20):
    return tf.contrib.layers.repeat(X,
                                    n_res,
                                    residual_block,
                                    scope = 'blocks')

def residual_block(X,
                   scope,
                   f = (1, 1, 1, 3, 3, 3, 3),
                   filters = (32, 32, 32, 64, 64, 64, 32),
                   strides = (1, 1, 1, 1, 1, 1, 1),
                   paddings = ('valid', 'valid', 'valid', 'same', 'same', 'same', 'same'),
                   names = ('_a', '_b', '_c', '_d', '_e', '_f', '_g'),
                   seed = 0,
                   activation_func = 'relu'):

    conv_name_base = 'res_' + scope + '_branch'

    F_a, F_b, F_c, F_d, F_e, F_f, F_g = filters
    f_a, f_b, f_c, f_d, f_e, f_f, f_g = f
    s_a, s_b, s_c, s_d, s_e, s_f, s_g = strides
    p_a, p_b, p_c, p_d, p_e, p_f, p_g = paddings
    n_a, n_b, n_c, n_d, n_e, n_f, n_g = names

    X_A = X

    X_B = Activation(activation_func)(X)

    X_B1 = Conv2D(filters = F_a,
                  kernel_size = (f_a, f_a),
                  strides = (s_a, s_a),
                  padding = p_a,
                  name = conv_name_base + n_a,
                  kernel_initializer = glorot_uniform(seed = seed))(X_B)

    X_B2 = Conv2D(filters = F_b,
                  kernel_size = (f_b, f_b),
                  strides = s_b,
                  padding = p_b,
                  name = conv_name_base + n_b,
                  kernel_initializer = glorot_uniform(seed = seed))(X_B)
    X_B2 = Activation(activation_func)(X_B2)
    X_B2 = Conv2D(filters = F_d,
                  kernel_size = (f_d, f_d),
                  strides = (s_d, s_d),
                  padding = p_d,
                  name = conv_name_base + n_d,
                  kernel_initializer = glorot_uniform(seed = seed))(X_B2)

    X_B3 = Conv2D(filters = F_c,
                  kernel_size = (f_c, f_c),
                  strides = s_c,
                  padding = p_c,
                  name = conv_name_base + n_c,
                  kernel_initializer = glorot_uniform(seed = seed))(X_B)
    X_B3 = Activation(activation_func)(X_B3)
    X_B3 = Conv2D(filters = F_e,
                  kernel_size = (f_e, f_e),
                  strides = s_e,
                  padding = p_e,
                  name = conv_name_base + n_e,
                  kernel_initializer = glorot_uniform(seed = seed))(X_B3)
    X_B3 = Activation(activation_func)(X_B3)
    X_B3 = Conv2D(filters = F_f,
                  kernel_size = (f_f, f_f),
                  strides = s_f,
                  padding = p_f,
                  name = conv_name_base + n_f,
                  kernel_initializer = glorot_uniform(seed = seed))(X_B3)

    X_B_concat = tf.concat([X_B1, X_B2, X_B3], axis = -1)
    X_B_concat = Activation(activation_func)(X_B_concat)
    X_B_concat = Conv2D(filters = F_g,
                        kernel_size = (f_g, f_g),
                        strides = (s_g, s_g),
                        padding = p_e,
                        name = conv_name_base + n_g,
                        kernel_initializer = glorot_uniform(seed = seed))(X_B_concat)
    X_out = Add()([X_A, X_B_concat * 0.3])

    return X_out


def identity_block(X, f, filters, stage, block):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters = F1,
               kernel_size = (1, 1),
               strides = (1,1),
               padding = 'valid',
               name = conv_name_base + '2a',
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F2,
               kernel_size = (f, f),
               strides = (1,1),
               padding = 'same',
               name = conv_name_base + '2b',
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F3,
               kernel_size = (1, 1),
               strides = (1,1),
               padding = 'valid',
               name = conv_name_base + '2c',
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X


def convolutional_block(X, f, filters, stage, block, s = 2):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(F1,
               (1, 1),
               strides = (s,s),
               padding = 'valid',
               name = conv_name_base + '2a',
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(F2,
               (f,f),
               strides = (1,1),
               padding = 'same',
               name = conv_name_base + '2b',
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(F3,
               (1,1),
               strides = (1,1),
               padding = 'valid',
               name = conv_name_base + '2c',
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    X_shortcut = Conv2D(F3,
                        (1,1),
                        strides = (s,s),
                        padding = 'valid',
                        name = conv_name_base + 'l',
                        kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + 'l')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    X = convolutional_block(X, f = 3, filters = [128,128,512], stage = 3, block = 'a', s = 2)
    X = identity_block(X, 3, [128,128,512], stage=3, block='b')
    X = identity_block(X, 3, [128,128,512], stage=3, block='c')
    X = identity_block(X, 3, [128,128,512], stage=3, block='d')

    X = convolutional_block(X, f = 3, filters = [256,256,1024], s = 2, block = 'a', stage = 4)
    X = identity_block(X, 3, [256,256,1024], stage=4, block='b')
    X = identity_block(X, 3, [256,256,1024], stage=4, block='c')
    X = identity_block(X, 3, [256,256,1024], stage=4, block='d')
    X = identity_block(X, 3, [256,256,1024], stage=4, block='e')
    X = identity_block(X, 3, [256,256,1024], stage=4, block='f')

    X = convolutional_block(X, f = 3, filters = [512,512,2048], s = 2, block = 'a', stage = 5)
    X = identity_block(X, 3, [512,512,2048], stage=5, block='b')
    X = identity_block(X, 3, [512,512,2048], stage=5, block='c')

    X = AveragePooling2D(pool_size=(2, 2), name = 'avg_pool')(X)

    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model