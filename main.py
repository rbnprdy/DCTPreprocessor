# main.py
#
# The entrypoint for the Discrete Cosine Transform Preprocessor.
#
# Created by Ruben Purdy and Kray Althaus for ECE429 at The University of
# Arizona.

import argparse

from trainer import train_session_fc, train_session_conv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='The deep learning model to use.',
                        choices=['fc', 'conv'])
    parser.add_argument('dataset', help='The dataset to use.',
                        choices=['mnist'])
    parser.add_argument('percentage', help='The percentage of dct components'
                        + ' to keep.', type=float)
    parser.add_argument('-s', '--stride', help='The stride to use when'
                        + ' perfoming the blockwise dct.', type=int,
                        default=4)
    parser.add_argument('-e', '--epochs', help='The number of epochs to train'
                        + ' for.', type=int, default=10)
    parser.add_argument('-b', '--batch_size', help='The batch size to use for'
                        + ' training.', type=int, default=128)
    parser.add_argument('-v', '--verbosity', help='The verbosity level.',
                        choices=[0, 1], default=0, type=int)
    args = parser.parse_args()

    if args.model == 'fc':
        train_session_fc(args.percentage, batch_size=args.batch_size,
                         epochs=args.epochs, stride=args.stride,
                         verbose=args.verbosity)
    else:
        train_session_conv(args.percentage, batch_size=args.batch_size,
                           epochs=args.epochs, stride=args.stride,
                           verbose=args.verbosity)


if __name__ == "__main__":
    main()
