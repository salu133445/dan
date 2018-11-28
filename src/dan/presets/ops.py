"""Tensorflow ops."""
import tensorflow as tf

CONV_KERNEL_INITIALIZER = tf.truncated_normal_initializer(stddev=0.05)
DENSE_KERNEL_INITIALIZER = tf.truncated_normal_initializer(stddev=0.05)

dense = lambda i, u: tf.layers.dense(
    i, u, kernel_initializer=DENSE_KERNEL_INITIALIZER)
conv2d = lambda i, f, k, s: tf.layers.conv2d(
    i, f, k, s, kernel_initializer=CONV_KERNEL_INITIALIZER)
conv3d = lambda i, f, k, s: tf.layers.conv3d(
    i, f, k, s, kernel_initializer=CONV_KERNEL_INITIALIZER)
tconv2d = lambda i, f, k, s: tf.layers.conv2d_transpose(
    i, f, k, s, kernel_initializer=CONV_KERNEL_INITIALIZER)
tconv3d = lambda i, f, k, s: tf.layers.conv3d_transpose(
    i, f, k, s, kernel_initializer=CONV_KERNEL_INITIALIZER)

maxpool2d = tf.layers.max_pooling2d
maxpool3d = tf.layers.max_pooling3d
avgpool2d = tf.layers.average_pooling2d
avgpool3d = tf.layers.average_pooling3d

flatten = tf.layers.flatten

def get_normalization(norm_type, training=None):
    """Return the normalization function."""
    if norm_type == 'batch_norm':
        return lambda x: tf.layers.batch_normalization(x, training=training)
    if norm_type == 'layer_norm':
        return tf.contrib.layers.layer_norm
    if norm_type is None or norm_type == '':
        return lambda x: x
    raise ValueError("Unrecognizable normalization type.")

def convconcat(tensor_in, condition, reshape_shape):
    """Concatenate conditions to a tensor for convolutional layers."""
    reshaped = tf.reshape(condition, reshape_shape)
    out_shape = (
        [tf.shape(tensor_in)[0]] + tensor_in.get_shape().as_list()[1:-1] +
        [condition.get_shape().as_list()[1]])
    to_concat = reshaped * tf.ones(out_shape)
    return tf.concat([tensor_in, to_concat], -1)

concat2d = lambda t, c: convconcat(t, c, (-1, 1, 1, c.get_shape()[1]))
concat3d = lambda t, c: convconcat(t, c, (-1, 1, 1, 1, c.get_shape()[1]))
