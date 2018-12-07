# preprocessor.py
#
# Preprocessing functions for reducing data size using the Discrete Cosine
# Transform.
#
# Created by Ruben Purdy and Kray Althaus for ECE429 at the University of
# Arizona

from __future__ import print_function

import numpy as np
from keras import backend as K
from dct import dct_set_gray, dct_set_color


def get_clip_index_fc(data, percentage):
    """Returns array that can be used to keep `percentage` percent of the
    elements from `data`"""
    val = np.sort(data)[int(len(data)*(1-percentage/100))]
    return np.where(data > val)


def get_clip_index_conv(data, percentage):
    """Returns an array that can be used to set values to keep `percentage`
    percent of the values in `data` as possibly non-zero"""
    original_shape = data.shape
    data = data.reshape(np.multiply.reduce(data.shape))
    val = np.sort(data)[int(len(data)*(1-percentage/100))]
    data = data.reshape(original_shape)
    return np.where(data < val)


def preprocess_fc(x_train, x_test, percentage, stride=8, verbose=0):
    """Keeps `percentage` percent of the DCT components from input image
    data. Returns data as DCT components."""
    x_train = dct_set_gray(x_train, stride, verbose=verbose)
    x_test = dct_set_gray(x_test, stride, verbose=verbose)

    x_train = x_train.reshape(x_train.shape[0],
                              x_train.shape[1]*x_train.shape[2])

    x_test = x_test.reshape(x_test.shape[0],
                            x_test.shape[1]*x_test.shape[2])

    variances = np.var(x_train, axis=0)
    stdevs = np.sqrt(variances)
    idx = get_clip_index_fc(stdevs, percentage)
    x_train_clipped = np.zeros((x_train.shape[0], len(idx[0])))
    x_test_clipped = np.zeros((x_test.shape[0], len(idx[0])))
    for i, image in enumerate(x_train):
        x_train_clipped[i] = image[idx]
    for i, image in enumerate(x_test):
        x_test_clipped[i] = image[idx]

    return x_train_clipped, x_test_clipped


def preprocess_conv(x_train, x_test, percentage, stride, grayscale=True,
                    verbose=0):
    """Keeps `percentage` percent of the DCT components from input image data.
    The rest of the components are set to 0. Returns data as DCT
    compontents."""
    if grayscale:
        x_train = dct_set_gray(x_train, stride, verbose=verbose)
        x_test = dct_set_gray(x_test, stride, verbose=verbose)

        variances = np.var(x_train, axis=0)
        stdevs = np.sqrt(variances)
        idx = get_clip_index_conv(stdevs, percentage)
        for image in x_train:
            image[idx] = 0
        for image in x_test:
            image[idx] = 0
    else:
        x_train = dct_set_color(x_train, stride, verbose=verbose)
        x_test = dct_set_color(x_test, stride, verbose=verbose)
        if K.image_data_format() == 'channels_first':
            for i in range(3):
                variances = np.var(x_train[:, i, :, :], axis=0)
                stdevs = np.sqrt(variances)
                idx = get_clip_index_conv(stdevs, percentage)
                for image in x_train:
                    channel = image[i]
                    channel[idx] = 0
                for image in x_test:
                    channel = image[i]
                    channel[idx] = 0
        else:
            for i in range(3):
                variances = np.var(x_train[:, :, :, i], axis=0)
                stdevs = np.sqrt(variances)
                idx = get_clip_index_conv(stdevs, percentage)
                for image in x_train:
                    channel = image[:, :, i]
                    channel[idx] = 0
                for image in x_test:
                    channel = image[:, :, i]
                    channel[idx] = 0

    return x_train, x_test
