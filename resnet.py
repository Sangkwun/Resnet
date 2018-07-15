from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D



def resnet(input_shape = None):

    def conv2_block(input_layer):
        pooling = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(input_layer)

    if input_shape is None:
        _shape = (224, 224, 3)
    else:
        _shape = input_shape

    # 224 x 224 x 3
    inputs = Input(shape=_shape, name="input_layer")

    # 112 x 112 x 64
    conv1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')(inputs)


resnet()