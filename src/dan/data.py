"""This file contains functions for loading and preprocessing the training data.
"""
import logging
import numpy as np
import tensorflow as tf
from dan.config import SHUFFLE_BUFFER_SIZE, PREFETCH_SIZE
LOGGER = logging.getLogger(__name__)

# --- Data loader --------------------------------------------------------------
def load_data_from_npy(filename):
    """Load and return the training data from a npy file."""
    return np.load(filename)

def load_data(data_source, filename):
    """Load and return the training data."""
    if data_source == 'sa':
        import SharedArray as sa
        return sa.attach(filename)
    if data_source == 'npy':
        return load_data_from_npy(filename)
    raise ValueError("Expect `data_source` to be one of 'sa' and 'npy'. But "
                     "get " + str(data_source))

# --- Dataset Utilities --------------------------------------------------------
def _gen_data(data, labels):
    """Yield data and label pairs."""
    for i, item in enumerate(data):
        yield (item, [labels[i]])

def set_data_shape(data, data_shape):
    """Set the label shape and return the label."""
    data.set_shape(data_shape)
    return data

def set_label_shape(label):
    """Set the label shape and return the label."""
    label.set_shape([1])
    return label

# --- Tensorflow Dataset -------------------------------------------------------
def get_dataset(data, labels, batch_size, data_shape, n_classes, repeat=False):
    """Create  and return a tensorflow dataset from an array."""
    assert len(data) == len(labels), (
        "Lengths of `data` and `lables` do not match.")
    dataset = tf.data.Dataset.from_generator(
        lambda: _gen_data(data, labels), (tf.float32, tf.int32))
    dataset = dataset.map(lambda data, label: (
        set_data_shape(data, data_shape), set_label_shape(label)))
    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size).prefetch(PREFETCH_SIZE)
    return dataset
