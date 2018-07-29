import resnet

from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

batch_size = 256
input_shape = (32, 32, 3)
classes= 10

model = resnet.resnet(input_shape=input_shape, classes=classes, shortcut=False)
model.summary()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255.
x_test = x_test / 255.

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)
datagen.fit(x_train)
train_data = datagen.flow(
                        x_train,
                        to_categorical(y_train),
                        batch_size=batch_size,
                        shuffle=True)

tensorboard = TensorBoard(log_dir='./graph/without_shortconnection',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=False)

chk_point = ModelCheckpoint('./weights/weights.best.hdf5',
                            monitor='val_loss',
                            verbose=0,
                            save_best_only=True,
                            save_weights_only=False,
                            mode='auto',
                            period=1)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit_generator(train_data,
                    steps_per_epoch= x_train.shape[0] // batch_size,
                    epochs=185,
                    verbose=1,
                    validation_data=(x_test[:100], to_categorical(y_test[:100])),
                    callbacks=[tensorboard, chk_point])

# Evaluate the model
scores = model.evaluate(x_test / 255.0, to_categorical(y_test))
print(scores)