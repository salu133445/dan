"""This file defines the network architecture for the discriminator."""
import tensorflow as tf
from tensorflow.nn import relu, leaky_relu, tanh, sigmoid
from ..ops import dense, conv2d, maxpool2d, flatten, concat2d, get_normalization

NORMALIZATION = None # 'batch_norm', 'layer_norm', None
ACTIVATION = relu # relu, leaky_relu, tanh, sigmoid

class Discriminator:
    """Class that defines the discriminator."""
    def __init__(self, name='Discriminator'):
        self.name = name

    def __call__(self, tensor_in, condition=None, training=None):
        norm = get_normalization(NORMALIZATION, training)
        conv_layer = lambda i, f, k, s: ACTIVATION(norm(conv2d(
            concat2d(i, condition), f, k, s)))
        dense_layer = lambda i, u: ACTIVATION(norm(dense(
            tf.concat((i, condition), -1), u)))

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            h = tensor_in
            h = conv_layer(h, 32, (3, 3), (3, 3))
            h = conv_layer(h, 64, (3, 3), (3, 3))
            h = maxpool2d(h, (2, 2), (2, 2))
            h = flatten(h)
            h = dense_layer(h, 128)
            h = dense(h, 1)

        return h
