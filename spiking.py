from __future__ import print_function

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import backend as K
from tealayers import Tea, AdditivePooling
from preprocessor import preprocess

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train_session(i, batch_size=128, epochs=100, num_classes=10):
    (x_train, y_train), (x_test, y_test) = preprocess(i)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    early_stop = EarlyStopping(patience=20)

    print('Training with {} parameters ({}%).'.format(
        x_train.shape[1], i))

    layers = [[64, 64, 64, 64], [250]]
    kernels = [[16, 16], 256]
    strides = [[12, 12], 256]
    network = zip(layers, kernels, strides)

    model = Sequential()
    for num, (layer, kernel, stride) in enumerate(network):
        model.add(Tea(layer, kernel, stride, activation='sigmoid'))
        if num != len(layer):
            model.add(Dropout(0.2))
            model.add(AdditivePooling(10))
            model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                                optimizer=Adam(),
                                metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[early_stop],
                        validation_split=0.2)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('\tTest accuracy:', score[1])

#train_session(100)
#quit()
#for i in range(1, 10):
#    train_session(i)

for i in range(10, 100, 10):
    train_session(i)
