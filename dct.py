# dct.py
#
# Discrete Cosine Transform functions for use preprocessing images.
#
# Created by Ruben Purdy and Kray Althaus for ECE429 at The University of
# Arizona

import numpy as np
from keras import backend as K
from scipy import fftpack

def dct_2d(img):
    """Performs a 2D DCT transform on the input image"""
    return fftpack.dct(
        fftpack.dct(img, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct_2d(img):
    """Performs a 2D Inverse DCT transform on the input image"""
    return fftpack.idct(
        fftpack.idct(img, axis=0, norm='ortho'), axis=1, norm='ortho')


def dct_image(im, stride=4):
    """Perfomrs a 2D DCT on an image in blocks of size `stride`."""
    dct = np.zeros(im.shape)
    for i in range(0, im.shape[0], stride):
        for j in range(0, im.shape[1], stride):
            dct[i:(i+stride), j:(j+stride)] = \
                dct_2d(im[i:(i+stride), j:(j+stride)])
    return dct


def idct_image(im, stride=4):
    """Perfomrs a 2D Inverse DCT on an image in blocks of size `stride`."""
    idct = np.zeros(im.shape)
    for i in range(0, im.shape[0], stride):
        for j in range(0, im.shape[1], stride):
            idct[i:(i+stride), j:(j+stride)] = \
                idct_2d(im[i:(i+stride), j:(j+stride)])
    return idct


def dct_set_gray(data, stride, verbose=0):
    """Performs a dct on a set of grayscale images with blocks of size
    `stide`."""
    dct = np.zeros(data.shape)
    for i in range(len(data)):
        if verbose:
            print('\t{}/{} images'.format(i + 1, len(data)), end='\r',
                  flush=True)
        dct[i] = dct_image(data[i], stride=stride)
    if verbose:
        print()
    return dct


def dct_set_color(data, stride, verbose=0):
    """Performs ad dct on a set of color images with blocks of size `stride`.
    """
    if K.image_data_format() == 'channels_first':
        for i in range(len(data)):
            if verbose:
                print('\t{}/{} images'.format(i + 1, len(data)), end='\r',
                      flush=True)
            for j in range(3):
                data[i, j, :, :] = dct_image(data[i, j, :, :], stride=stride)
    else:
        for i in range(len(data)):
            if verbose:
                print('\t{}/{} images'.format(i + 1, len(data)), end='\r',
                      flush=True)
            for j in range(3):
                data[i, :, :, j] = dct_image(data[i, :, :, j], stride=stride)

    if verbose:
        print()
    return data
