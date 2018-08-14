from src.drssrn_symbol import residual_block
import keras.backend as K


model = residual_block(input_shape=FIT_INPUT_SHAPE, classes = 6)

K.get_session().graph