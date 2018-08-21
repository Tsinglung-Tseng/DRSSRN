

class ResBlock:
    def __init__(self):
        self.x_a = tf.layers.conv2d(inputs,
                                    filters,
                                    kernel_size,
                                    strides=(1, 1),
                                    padding='valid',
                                    data_format='channels_last',
                                    dilation_rate=(1, 1),
                                    activation=None,
                                    use_bias=True,
                                    kernel_initializer=None,
                                    bias_initializer=tf.zeros_initializer(),
                                    kernel_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    trainable=True,
                                    name=None,
                                    reuse=None)

class SRTrys(Model):

    # TODO


g = SRTrys('g')
g.make()