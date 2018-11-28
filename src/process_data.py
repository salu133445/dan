"""Load and save the MNIST dataset to shared memory via SharedArray package."""
import os
import argparse
import numpy as np
import SharedArray as sa

def parse_arguments():
    """Parse and return the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root', help="Root directory of the dataset.")
    args = parser.parse_args()
    return args.dataset_root

def save_to_sa(name, data):
    """Save data to SharedArray."""
    arr = sa.create(name, data.shape, data.dtype)
    np.copyto(arr, data)

def process(dataset_root):
    """Load and save the dataset to SharedArray."""
    with open(os.path.join(dataset_root, 'train-images-idx3-ubyte')) as file:
        loaded = np.fromfile(file=file, dtype=np.uint8)
        save_to_sa('mnist_train_x', loaded[16:].reshape((60000, 28, 28, 1)))

    with open(os.path.join(dataset_root, 't10k-images-idx3-ubyte')) as file:
        loaded = np.fromfile(file=file, dtype=np.uint8)
        save_to_sa('mnist_val_x', loaded[16:].reshape((10000, 28, 28, 1)))

    with open(os.path.join(dataset_root, 'train-labels-idx1-ubyte')) as file:
        loaded = np.fromfile(file=file, dtype=np.uint8)
        save_to_sa('mnist_train_y', loaded[8:].reshape((60000)))

    with open(os.path.join(dataset_root, 't10k-labels-idx1-ubyte')) as file:
        loaded = np.fromfile(file=file, dtype=np.uint8)
        save_to_sa('mnist_val_y', loaded[8:].reshape((10000)))

def main():
    """Main function"""
    dataset_root = parse_arguments()
    process(dataset_root)

if __name__ == '__main__':
    main()
