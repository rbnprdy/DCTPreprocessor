from __future__ import print_function

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import backend as K
from preprocessor import preprocess

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train_session(i, batch_size=128, epochs=100, num_classes=10):
    (x_train, y_train), (x_test, y_test) = preprocess(i, stride=4)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    early_stop = EarlyStopping(patience=20)

    print('Input shape: {} ({}% of parameters).'.format(
        x_train.shape, i))
    """
    img_rows, img_cols = 28, 28
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                                      activation='relu',
                                      input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    """
    model = Sequential()
    model.add(Dense(x_train.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    #model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                                optimizer=Adam(),
                                metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        callbacks=[early_stop],
                        validation_split=0.2)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('\tTest accuracy:', score[1])

for i in range(10, 100, 10):
    train_session(i)
