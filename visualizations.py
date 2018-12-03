# visualizations.py
#
# Some functions to help visualize the effects of the Discrete Cosine
# Transform.
#
# Created by Ruben Purdy and Kray Althaus for ECE429 at The University of
# Arizona.

import argparse

import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from preprocessor import preprocess_conv
from dct import dct_image, idct_image, dct_set_gray


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('percentage', help='The percentage of dct components'
                        + ' to keep.', type=float)
    parser.add_argument('stride', help='The stride to use for the blockwise'
                        + ' dct.', type=int)
    args = parser.parse_args()

    (x_train, _), (x_test, _) = mnist.load_data()
    x_train_pre = np.copy(x_train)
    x_test_pre = np.copy(x_test)
    x_train, x_test = preprocess_conv(x_train, x_test, args.percentage,
                                              args.stride, verbose=1)

    fig, axes = plt.subplots(3, 3)
    for i in range(3):
        axes[i, 0].imshow(x_train_pre[i], cmap='gray')
        axes[i, 0].set_title('Original Image')
        axes[i, 0].get_xaxis().set_visible(False)
        axes[i, 0].get_yaxis().set_visible(False)
        axes[i, 1].imshow(x_train[i], cmap='gray')
        axes[i, 1].set_title('DCT ({}%)'.format(args.percentage))
        axes[i, 1].get_xaxis().set_visible(False)
        axes[i, 1].get_yaxis().set_visible(False)
        axes[i, 2].imshow(idct_image(x_train[i], stride=args.stride),
                          cmap='gray')
        axes[i, 2].set_title('IDCT ({}%)'.format(args.percentage))
        axes[i, 2].get_xaxis().set_visible(False)
        axes[i, 2].get_yaxis().set_visible(False)

    plt.show()


if __name__ == "__main__":
    main()
