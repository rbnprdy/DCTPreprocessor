from __future__ import print_function

import numpy as np
from keras.datasets import mnist
from scipy import fftpack
import matplotlib.pyplot as plt

def dct_2d(img):
    return fftpack.dct(
        fftpack.dct(img, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct_2d(img):
    return fftpack.idct(
        fftpack.idct(img, axis=0, norm='ortho'), axis=1, norm='ortho')

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def dct_image(im, stride=8):
    dct = np.zeros(im.shape)
    for i in range(0,im.shape[0],stride):
        for j in range(0, im.shape[1],stride):
            dct[i:(i+stride),j:(j+stride)] = \
                dct_2d(im[i:(i+stride),j:(j+stride)])
    return dct

def idct_image(im):
    idct = np.zeros(im.shape)
    for i in range(0,im.shape[0],8):
        for j in range(0, im.shape[1],8):
            idct[i:(i+8),j:(j+8)] = idct_2d(im[i:(i+8),j:(j+8)])
    return idct

def dct_set(data, stride):
    for i in range(len(data)):
        data[i] = dct_image(data[i], stride=stride)
    data = data / np.max(data)
    return data

def clip(data, percentage):
    # Get value to clip by
    val = np.sort(data)[int(len(data)*(1-percentage/100))]
    return np.where(data > val)

def preprocess(percentage, stride=8):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = dct_set(x_train, stride)
    x_test = dct_set(x_test, stride)

    x_train = x_train.reshape(x_train.shape[0],
                              x_train.shape[1]*x_train.shape[2])

    x_test = x_test.reshape(x_test.shape[0],
                            x_test.shape[1]*x_test.shape[2])

    variances = np.var(x_train, axis=0)
    stdevs = np.sqrt(variances)
    idx = clip(stdevs, percentage)

    x_train_clipped = np.squeeze(x_train[:,idx])
    x_test_clipped = np.squeeze(x_test[:,idx])

    return (x_train_clipped, y_train), (x_test_clipped, y_test)

