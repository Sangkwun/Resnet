import resnet

from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, TensorBoard

model = resnet.resnet(input_shape=(32, 32, 3), classes=10)
model.summary()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
tensorboard = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

# Compile the model
model.compile(loss='categorical_crossentropy',
                optimizer=Adam(lr=0.0001, decay=1e-6),
                metrics=['accuracy'])

# Train the model
model.fit(x_train / 255.0, to_categorical(y_train),
            batch_size=128,
            shuffle=True,
            epochs=250,
            validation_data=(x_test / 255.0, to_categorical(y_test)),
            callbacks=[EarlyStopping(min_delta=0.001, patience=3), tensorboard])

# Evaluate the model
scores = model.evaluate(x_test / 255.0, to_categorical(y_test))