"""This file implements the spectral normalization."""
import tensorflow as tf
from .ops import CONV_KERNEL_INITIALIZER, DENSE_KERNEL_INITIALIZER

def spectral_normalize(w, n_iters=1, name='spectral_norm'):
    """
    Return spectral normalized weights.

    Parameters
    ----------
    w : `tf.Tensor`
        The input weight tensor.
    n_iters : int
        The number of power iterations to run. Defaults to 1.
    name : str
        Name of the operation.

    Returns
    -------
    Spectral normalized weight tensor.
    """
    with tf.variable_scope(name):
        w_reshaped = tf.reshape(w, [-1, w.get_shape()[-1]])
        u = tf.get_variable(
            'u', [1, w.get_shape()[-1]],
            initializer=tf.truncated_normal_initializer(), trainable=False)

        def power_iteration(i, u_i, v_i):
            """Apply power iteration to estimate the spectral norm."""
            v_ip1 = tf.nn.l2_normalize(tf.matmul(u_i, tf.transpose(w_reshaped)))
            u_ip1 = tf.nn.l2_normalize(tf.matmul(v_ip1, w_reshaped))
            return i + 1, u_ip1, v_ip1

        _, u_final, v_final = tf.while_loop(
            lambda i, _1, _2: i < n_iters,
            power_iteration,
            (tf.constant(0, dtype=tf.int32), u, tf.zeros(
                dtype=tf.float32, shape=[1, w_reshaped.get_shape()[0]])))

        sigma = tf.matmul(
            tf.matmul(v_final, w_reshaped), tf.transpose(u_final))[0, 0]
        w_bar = w_reshaped / sigma
        with tf.control_dependencies([u.assign(u_final)]):
            w_norm = tf.reshape(w_bar, w.get_shape())

    return w_norm

def sn_dense(inputs, units, name='sn_dense'):
    with tf.variable_scope(name) as scope:
        weight = tf.get_variable(
            'w', [inputs.get_shape()[1], units], tf.float32,
            initializer=DENSE_KERNEL_INITIALIZER)
        bias = tf.get_variable('b', [units], initializer=tf.zeros_initializer())
        return tf.matmul(inputs, spectral_normalize(weight)) + bias

def sn_conv2d(inputs, filters, kernel_size, strides=(1, 1), padding='VALID',
              name='sn_conv2d'):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    with tf.variable_scope(name) as scope:
        weight = tf.get_variable(
            'w', [kernel_size[0], kernel_size[1], inputs.get_shape()[-1],
                  filters],
            initializer=CONV_KERNEL_INITIALIZER)
        conv = tf.nn.conv2d(
            inputs, spectral_normalize(weight),
            strides=[1, strides[0], strides[1], 1], padding=padding)
        biases = tf.get_variable(
            'b', [filters], initializer=tf.zeros_initializer())
        return tf.reshape(
            tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])

def sn_conv3d(inputs, filters, kernel_size, strides=(1, 1, 1), padding='VALID',
              name='sn_conv2d'):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(strides, int):
        strides = (strides, strides, strides)
    with tf.variable_scope(name) as scope:
        weight = tf.get_variable(
            'w', [kernel_size[0], kernel_size[1], kernel_size[2],
                  inputs.get_shape()[-1], filters],
            initializer=CONV_KERNEL_INITIALIZER)
        conv = tf.nn.conv3d(
            inputs, spectral_normalize(weight),
            strides=[1, strides[0], strides[1], strides[2], 1], padding=padding)
        biases = tf.get_variable(
            'b', [filters], initializer=tf.zeros_initializer())
        return tf.reshape(
            tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])
