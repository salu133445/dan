"""This file processes and saves the MNIST dataset to shared memory via
SharedArray package. This file also creates the imbalanced and the very
imbalanced datasets."""
import os
import argparse
import numpy as np
import SharedArray as sa

def parse_arguments():
    """Parse and return the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root', help="Root directory of the dataset.")
    parser.add_argument('--make_imbalanced', action='store_true', default=False)
    parser.add_argument('--make_super_imbalanced', action='store_true',
                        default=False)
    args = parser.parse_args()
    return args.dataset_root, args.make_imbalanced, args.make_super_imbalanced

def save_to_sa(name, data):
    """Save data to SharedArray."""
    arr = sa.create(name, data.shape, data.dtype)
    np.copyto(arr, data)

def shift(data, shift_x, shift_y):
    """Shift the data."""
    slice_x = slice(-shift_x) if shift_x > 0 else slice(-shift_x, None)
    slice_y = slice(-shift_y) if shift_y > 0 else slice(-shift_y, None)
    pad_x = (shift_x, 0) if shift_x > 0 else (0, -shift_x)
    pad_y = (shift_y, 0) if shift_y > 0 else (0, -shift_y)
    shifted = np.pad(
        data[:, slice_x, slice_y], ((0, 0), pad_x, pad_y, (0, 0)), 'constant')
    return shifted

def process(dataset_root, make_imbalanced=False, make_super_imbalanced=False):
    """Load and save the dataset to SharedArray."""
    with open(os.path.join(dataset_root, 'train-images-idx3-ubyte')) as file:
        loaded = np.fromfile(file=file, dtype=np.uint8)
        train_x = loaded[16:].reshape((60000, 28, 28, 1))
        save_to_sa('mnist_train_x', train_x)

    with open(os.path.join(dataset_root, 'train-labels-idx1-ubyte')) as file:
        loaded = np.fromfile(file=file, dtype=np.uint8)
        train_y = loaded[8:].reshape((60000))
        save_to_sa('mnist_train_y', train_y)

    if make_imbalanced:
        indices_zero = (train_y == 0).nonzero()[0]
        unshifted = train_x[indices_zero]
        imbalanced_train_x = np.concatenate(
            (unshifted, shift(unshifted, 1, 0), shift(unshifted, 0, 1),
             shift(unshifted, -1, 0), shift(unshifted, 0, -1)), 0)
        indices_others = np.random.choice(
            (train_y > 0).nonzero()[0], 60000 - 5 * len(indices_zero), False)
        imbalanced_train_x = np.concatenate(
            (imbalanced_train_x, train_x[indices_others]), 0)
        imbalanced_train_y = np.concatenate(
            (np.zeros(5 * len(indices_zero), np.uint8),
             train_y[indices_others]), 0)
        permutation = np.random.permutation(60000)
        save_to_sa('imbalanced_mnist_train_x', imbalanced_train_x[permutation])
        save_to_sa('imbalanced_mnist_train_y', imbalanced_train_y[permutation])

    if make_super_imbalanced:
        indices_zero = (train_y == 0).nonzero()[0]
        unshifted = train_x[indices_zero]
        super_imbalanced_train_x = np.concatenate(
            (unshifted, shift(unshifted, 1, 0), shift(unshifted, 0, 1),
             shift(unshifted, -1, 0), shift(unshifted, 0, -1),
             shift(unshifted, 1, 1), shift(unshifted, -1, -1)), 0)
        indices_others = np.random.choice(
            (train_y > 0).nonzero()[0], 60000 - 7 * len(indices_zero), False)
        super_imbalanced_train_x = np.concatenate(
            (super_imbalanced_train_x, train_x[indices_others]), 0)
        super_imbalanced_train_y = np.concatenate(
            (np.zeros(7 * len(indices_zero), np.uint8),
             train_y[indices_others]), 0)
        permutation = np.random.permutation(60000)
        save_to_sa(
            'super_imbalanced_mnist_train_x',
            super_imbalanced_train_x[permutation])
        save_to_sa(
            'super_imbalanced_mnist_train_y',
            super_imbalanced_train_y[permutation])

    with open(os.path.join(dataset_root, 't10k-images-idx3-ubyte')) as file:
        loaded = np.fromfile(file=file, dtype=np.uint8)
        save_to_sa('mnist_val_x', loaded[16:].reshape((10000, 28, 28, 1)))

    with open(os.path.join(dataset_root, 't10k-labels-idx1-ubyte')) as file:
        loaded = np.fromfile(file=file, dtype=np.uint8)
        save_to_sa('mnist_val_y', loaded[8:].reshape((10000)))

def main():
    """Main function"""
    dataset_root, make_imbalanced, make_super_imbalanced = parse_arguments()
    process(dataset_root, make_imbalanced, make_super_imbalanced)

if __name__ == '__main__':
    main()
