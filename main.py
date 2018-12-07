# main.py
#
# The entrypoint for the Discrete Cosine Transform Preprocessor.
#
# Created by Ruben Purdy and Kray Althaus for ECE429 at The University of
# Arizona.

import argparse

from trainer import train_session_mnist_fc, train_session_mnist_conv
from trainer import train_session_cifar10_conv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='The deep learning model to use.',
                        choices=['mnist_fc', 'mnist_conv', 'cifar10'])
    parser.add_argument('percentage', help='The percentage of dct components'
                        + ' to keep.', type=float)
    parser.add_argument('-s', '--stride', help='The stride to use when'
                        + ' perfoming the blockwise dct.', type=int,
                        default=4)
    parser.add_argument('-e', '--epochs', help='The number of epochs to train'
                        + ' for.', type=int, default=15)
    parser.add_argument('-b', '--batch_size', help='The batch size to use for'
                        + ' training.', type=int, default=128)
    parser.add_argument('-v', '--verbosity', help='The verbosity level.',
                        choices=[0, 1], default=0, type=int)
    args = parser.parse_args()

    if args.model == 'mnist_fc':
        train_session_mnist_fc(args.percentage, batch_size=args.batch_size,
                               epochs=args.epochs, stride=args.stride,
                               verbose=args.verbosity)
    elif args.model == 'mnist_conv':
        train_session_mnist_conv(args.percentage, batch_size=args.batch_size,
                                 epochs=args.epochs, stride=args.stride,
                                 verbose=args.verbosity)
    elif args.model == 'cifar10':
        train_session_cifar10_conv(args.percentage, batch_size=args.batch_size,
                                   epochs=args.epochs, stride=args.stride,
                                   verbose=args.verbosity)


if __name__ == "__main__":
    main()
