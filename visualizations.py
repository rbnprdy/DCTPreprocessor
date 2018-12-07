# visualizations.py
#
# Some functions to help visualize the effects of the Discrete Cosine
# Transform.
#
# Created by Ruben Purdy and Kray Althaus for ECE429 at The University of
# Arizona.

import argparse

import numpy as np
from keras.datasets import mnist, cifar10
import matplotlib.pyplot as plt
from preprocessor import preprocess_conv
from dct import idct_image, idct_image_color


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset to use for'
                        + ' visualization.', choices=['mnist', 'cifar10'])
    parser.add_argument('percentage', help='The percentage of dct components'
                        + ' to keep.', type=float)
    parser.add_argument('stride', help='The stride to use for the blockwise'
                        + ' dct.', type=int)
    args = parser.parse_args()

    if args.dataset == 'mnist':
        (x_train, _), (x_test, _) = mnist.load_data()
        x_train_pre = np.copy(x_train)
        x_train, x_test = preprocess_conv(x_train, x_test, args.percentage,
                                          args.stride, verbose=1)
    elif args.dataset == 'cifar10':
        (x_train, _), (x_test, _) = cifar10.load_data()
        x_train_pre = np.copy(x_train)
        x_train, x_test = preprocess_conv(x_train, x_test, args.percentage,
                                          args.stride, grayscale=False,
                                          verbose=1)
    fig, axes = plt.subplots(3, 3)
    for i in range(3):
        if args.dataset == 'mnist':
            axes[i, 0].imshow(x_train_pre[i], cmap='gray')
        elif args.dataset == 'cifar10':
            axes[i, 0].imshow(x_train_pre[i])
        axes[i, 0].set_title('Original Image')
        axes[i, 0].get_xaxis().set_visible(False)
        axes[i, 0].get_yaxis().set_visible(False)
        if args.dataset == 'mnist':
            axes[i, 1].imshow(x_train[i], cmap='gray')
        elif args.dataset == 'cifar10':
            axes[i, 1].imshow((x_train[i] - np.min(x_train[i]))
                              / (np.max(x_train) - np.min(x_train)))
        axes[i, 1].set_title('DCT ({}%)'.format(args.percentage))
        axes[i, 1].get_xaxis().set_visible(False)
        axes[i, 1].get_yaxis().set_visible(False)
        if args.dataset == 'mnist':
            axes[i, 2].imshow(idct_image(x_train[i], stride=args.stride),
                              cmap='gray')
        elif args.dataset == 'cifar10':
            axes[i, 2].imshow(idct_image_color(x_train[i],
                                               stride=args.stride).astype(int))
        axes[i, 2].set_title('IDCT ({}%)'.format(args.percentage))
        axes[i, 2].get_xaxis().set_visible(False)
        axes[i, 2].get_yaxis().set_visible(False)

    plt.show()


if __name__ == "__main__":
    main()
