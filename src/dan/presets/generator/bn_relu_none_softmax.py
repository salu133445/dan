"""This file defines the network architecture for the generator."""
import tensorflow as tf
from tensorflow.nn import relu, leaky_relu, tanh, sigmoid, softmax
from ..ops import dense, conv2d, maxpool2d, flatten, concat2d, get_normalization

NORMALIZATION = 'batch_norm' # 'batch_norm', 'layer_norm', None
ACTIVATION = relu # relu, leaky_relu, tanh, sigmoid
OUT_NORMALIZATION = None # 'batch_norm', 'layer_norm', None
OUT_ACTIVATION = softmax # relu, leaky_relu, tanh, sigmoid, softmax

class Generator:
    """Class that defines the generator."""
    def __init__(self, n_classes, name='Generator'):
        self.n_classes = n_classes
        self.name = name

    def __call__(self, tensor_in, condition=None, training=None):
        norm = get_normalization(NORMALIZATION, training)
        conv_layer = lambda i, f, k, s: ACTIVATION(norm(conv2d(i, f, k, s)))
        dense_layer = lambda i, u: ACTIVATION(norm(dense(i, u)))

        out_norm = get_normalization(OUT_NORMALIZATION, training)
        if OUT_ACTIVATION is None:
            out_dense_layer = lambda i, u: out_norm(dense(i, u))
        else:
            out_dense_layer = lambda i, u: OUT_ACTIVATION(out_norm(dense(i, u)))

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            h = tensor_in
            h = conv_layer(h, 32, (3, 3), (3, 3))
            h = conv_layer(h, 64, (3, 3), (3, 3))
            h = maxpool2d(h, (2, 2), (2, 2))
            h = flatten(h)
            h = dense_layer(h, 128)
            h = out_dense_layer(h, self.n_classes)

        return h
