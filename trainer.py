# trainer.py
#
# Helper functions for training FC and Convolutional networks
#
# Created by Ruben Purdy and Kray Althaus for ECE429 at the University of
# Arizona

from __future__ import print_function

import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import backend as K
from preprocessor import preprocess_fc, preprocess_conv
from models import FullyConnected, Convolution


def train_session_fc(percentage, batch_size=128, epochs=100, num_classes=10,
                     stride=4, verbose=0):

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = preprocess_fc(x_train, x_test, percentage, stride=stride,
                                    verbose=verbose)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train / np.max(x_train)
    x_test = x_test / np.max(x_test)

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    early_stop = EarlyStopping(patience=20)

    print('Input shape: {} (keeping {}% of DCT components.)'.format(
        x_train.shape, percentage))

    model = FullyConnected(x_train.shape[1], num_classes)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose,
              callbacks=[early_stop],
              validation_split=0.2)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('\tTest accuracy:', score[1])


def train_session_conv(percentage, batch_size=128, epochs=100, num_classes=10,
                       stride=4, img_rows=28, img_cols=28, verbose=0):

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = preprocess_conv(x_train, x_test, percentage,
                                      stride=stride, verbose=verbose)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train / np.max(x_train)
    x_test = x_test / np.max(x_test)

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    early_stop = EarlyStopping(patience=20)

    print('Input shape: {} (keeping {}% of DCT components.)'.format(
        x_train.shape, percentage))

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    model = Convolution(input_shape, num_classes)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose,
              callbacks=[early_stop],
              validation_split=0.2)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('\tTest accuracy:', score[1])
