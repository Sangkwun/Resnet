from keras.models import Model
from keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, BatchNormalization, add

def conv_block(input_layer,
                   kernel_size,
                   filters,
                   strides=(2,2),
                   first_strides=None,
                   short_con=True,
                   first_projection=True,
                   block_n=3,
                   name=None
                   ):

        filter_1, filter_2, filter_3 = filters
        bn_axis = 3

        if first_strides is None:
            first_strides = strides

        for i in range(1, block_n + 1):
            if i > 1:
                first_strides = (1, 1)

            # First conv layer of Block
            x = Conv2D(filter_1, kernel_size=(1,1), strides=first_strides, name="{}_{}_conv1".format(name, i))(input_layer)
            x = BatchNormalization(axis=bn_axis, name="{}_{}_bn1".format(name, i))(x)
            x = Activation('relu')(x)

            # Second conv layer of Block
            x = Conv2D(filter_2, kernel_size, strides=(1,1), padding='same', name="{}_{}_conv2".format(name, i))(x)
            x = BatchNormalization(axis=bn_axis, name="{}_{}_bn2".format(name, i))(x)
            x = Activation('relu')(x)

            # Third conv layer of Block
            x = Conv2D(filter_3, kernel_size=(1,1), name="{}_{}_conv3".format(name, i))(x)
            x = BatchNormalization(axis=bn_axis, name="{}_{}_bn3".format(name, i))(x)
            x = Activation('relu')(x)
            
            # shortcut part
            if short_con:
                if first_projection and i == 1:
                    # projection shortcut for extension
                    shortcut = Conv2D(filter_3, kernel_size=(1,1), strides=first_strides, name="{}_{}_short".format(name, i))(input_layer)
                    shortcut = BatchNormalization(axis=bn_axis, name="{}_{}_bn4".format(name, i))(shortcut)
                else:
                    # identity short cut
                    shortcut = input_layer

                x = add([x, shortcut], name="{}_{}_add".format(name, i))
            x = Activation('relu')(x)

            input_layer = x

        return x

def resnet(input_shape=None, shortcut=True):

    if input_shape is None:
        _shape = (224, 224, 3)
    else:
        _shape = input_shape
    
    # 224 x 224 x 3
    inputs = Input(shape=_shape, name="input_layer")

    # 112 x 112 x 64
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="stage2_0_maxpooling")(x)

    x = conv_block(input_layer=x, kernel_size=(3, 3), strides=(2, 2), first_strides=(1, 1), filters=[64, 64, 256], name="stage2", short_con=shortcut)
    x = conv_block(input_layer=x, kernel_size=(3, 3), strides=(2, 2), filters=[128, 128, 512], block_n=4, name="stage3", short_con=shortcut)
    x = conv_block(input_layer=x, kernel_size=(3, 3), strides=(2, 2), filters=[256, 256, 1024], block_n=6, name="stage4", short_con=shortcut)
    x = conv_block(input_layer=x, kernel_size=(3, 3), strides=(2, 2), filters=[512, 512, 2048], block_n=3, name="stage5", short_con=shortcut)

    model = Model(inputs=inputs, outputs=x)
    model.summary()


if __name__ == "__main__":
    resnet(shortcut=False)