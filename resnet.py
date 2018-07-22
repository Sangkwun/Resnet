from keras.models import Model
from keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, BatchNormalization, add

def conv_block(input_layer,
                   kernel_size,
                   filters,
                   strides=(2,2),
                   first_projection=True,
                   block_n=3,
                   name=None
                   ):

        filter_1, filter_2, filter_3 = filters
        bn_axis = 3
        
        for i in range(block_n):
            # First conv layer of Block
            x = Conv2D(filter_1, kernel_size=(1,1), strides=strides, name="{}_{}_conv1".format(name, i))(input_layer)
            x = BatchNormalization(axis=bn_axis, name="{}_1_bn".format(name))(x)
            x = layers.Activation('relu')(x)

            # Second conv layer of Block
            x = Conv2D(filter_2, kernel_size, strides=(1,1), padding='SAME', name="{}_{}_conv2".format(name, i))(x)
            x = BatchNormalization(axis=bn_axis, name="{}_2_bn".format(name))(x)
            x = layers.Activation('relu')(x)

            # Third conv layer of Block
            x = Conv2D(filter_3, kernel_size=(1,1), strides=(1,1), name="{}_{}_conv3".format(name, i))(x)
            x = BatchNormalization(axis=bn_axis, name="{}_3_bn".format(name))(x)
            x = layers.Activation('relu')(x)
            
            if first_projection and i = 0:
                # projection shortcut for extension
                short_con = Conv2D(filter_3, strides=strides, name="{}_{}_short".format(name, i))(input_layer)
                short_con = BatchNormalization(axis=bn_axis, name="{}_{}_bn".format(name, i))(short_con)
            else:
                # identity short cut
                short_con = input_layer

            x = add([x, shortcut])
            x = Activation('relu')(x)

            input_layer = x

        return x

def resnet(input_shape=None):

    if input_shape is None:
        _shape = (224, 224, 3)
    else:
        _shape = input_shape
    
    # 224 x 224 x 3
    inputs = Input(shape=_shape, name="input_layer")

    # 112 x 112 x 64
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')(inputs)
    x = BatchNormalization(axis=bn_axis, name="{}_1_bn".format(name))(x)
    x = layers.Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(conv1)



    model = Model(inputs=inputs, outputs=conv1)
    model.summary()


if __name__ == "__main__":
    resnet()