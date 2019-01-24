"""This file implements different adversarial losses."""
import tensorflow as tf

def get_adv_losses(discriminator_real_outputs, discriminator_fake_outputs,
                   kind):
    """Return the corresponding GAN losses for the generator and the
    discriminator."""
    if kind == 'classic':
        loss_fn = classic_gan_losses
    elif kind == 'nonsaturating':
        loss_fn = nonsaturating_gan_losses
    elif kind == 'imbalanced-nonsaturating-0.5':
        loss_fn = lambda r, f: imbalanced_nonsaturating_gan_losses(r, f, 0.5)
    elif kind == 'imbalanced-nonsaturating-0.9':
        loss_fn = lambda r, f: imbalanced_nonsaturating_gan_losses(r, f, 0.9)
    elif kind == 'imbalanced-nonsaturating-1.1':
        loss_fn = lambda r, f: imbalanced_nonsaturating_gan_losses(r, f, 1.1)
    elif kind == 'imbalanced-nonsaturating-2.0':
        loss_fn = lambda r, f: imbalanced_nonsaturating_gan_losses(r, f, 2.0)
    elif kind == 'wasserstein':
        loss_fn = wasserstein_gan_losses
    elif kind == 'imbalanced-wasserstein-0.5':
        loss_fn = lambda r, f: imbalanced_wasserstein_gan_losses(r, f, 0.5)
    elif kind == 'imbalanced-wasserstein-0.9':
        loss_fn = lambda r, f: imbalanced_wasserstein_gan_losses(r, f, 0.9)
    elif kind == 'imbalanced-wasserstein-1.1':
        loss_fn = lambda r, f: imbalanced_wasserstein_gan_losses(r, f, 1.1)
    elif kind == 'imbalanced-wasserstein-2.0':
        loss_fn = lambda r, f: imbalanced_wasserstein_gan_losses(r, f, 2.0)
    elif kind == 'hinge':
        loss_fn = hinge_gan_losses
    elif kind == 'imbalanced-hinge-0.5':
        loss_fn = lambda r, f: imbalanced_hinge_gan_losses(r, f, 0.5)
    elif kind == 'imbalanced-hinge-0.9':
        loss_fn = lambda r, f: imbalanced_hinge_gan_losses(r, f, 0.9)
    elif kind == 'imbalanced-hinge-1.1':
        loss_fn = lambda r, f: imbalanced_hinge_gan_losses(r, f, 1.1)
    elif kind == 'imbalanced-hinge-2.0':
        loss_fn = lambda r, f: imbalanced_hinge_gan_losses(r, f, 2.0)
    elif kind == 'least-squares':
        loss_fn = least_squares_gan_losses
    elif kind == 'absolute':
        loss_fn = absolute_gan_losses
    elif kind == 'new':
        loss_fn = new_gan_losses
    elif kind == 'relativistic-average':
        loss_fn = relativistic_average_gan_losses
    elif kind == 'relativistic-average-hinge':
        loss_fn = relativistic_average_hinge_gan_losses
    elif kind == 'improved-classic':
        loss_fn = improved_classic_gan_losses
    elif kind == 'improved-hinge':
        loss_fn = improved_hinge_gan_losses
    elif kind == 'improved-hinge-alternative':
        loss_fn = improved_hinge_alternative_gan_losses
    elif kind == 'nonsaturating-hinge':
        loss_fn = nonsaturating_hinge_gan_losses
    elif kind == 'minimax-hinge':
        loss_fn = minimax_hinge_gan_losses
    elif kind == 'double-absolute':
        loss_fn = double_absolute_gan_losses
    else:
        raise ValueError("Unrecognized adversarial loss type: " + str(kind))
    return loss_fn(discriminator_real_outputs, discriminator_fake_outputs)

def classic_gan_losses(discriminator_real_outputs, discriminator_fake_outputs):
    """Return the classic GAN losses for the generator and the discriminator.

    (Generator)      log(1 - sigmoid(D(G(z))))
    (Discriminator)  - log(sigmoid(D(x))) - log(1 - sigmoid(D(G(z))))
    """
    discriminator_loss_real = tf.losses.sigmoid_cross_entropy(
        tf.ones_like(discriminator_real_outputs), discriminator_real_outputs)
    discriminator_loss_fake = tf.losses.sigmoid_cross_entropy(
        tf.zeros_like(discriminator_fake_outputs), discriminator_fake_outputs)
    discriminator_loss = discriminator_loss_real + discriminator_loss_fake
    generator_loss = -discriminator_loss
    return generator_loss, discriminator_loss

def nonsaturating_gan_losses(discriminator_real_outputs,
                             discriminator_fake_outputs):
    """Return the non-saturating GAN losses for the generator and the
    discriminator.

    (Generator)      -log(sigmoid(D(G(z))))
    (Discriminator)  -log(sigmoid(D(x))) - log(1 - sigmoid(D(G(z))))
    """
    discriminator_loss_real = tf.losses.sigmoid_cross_entropy(
        tf.ones_like(discriminator_real_outputs), discriminator_real_outputs)
    discriminator_loss_fake = tf.losses.sigmoid_cross_entropy(
        tf.zeros_like(discriminator_fake_outputs), discriminator_fake_outputs)
    discriminator_loss = discriminator_loss_real + discriminator_loss_fake
    generator_loss = tf.losses.sigmoid_cross_entropy(
        tf.ones_like(discriminator_fake_outputs), discriminator_fake_outputs)
    return generator_loss, discriminator_loss

def imbalanced_nonsaturating_gan_losses(discriminator_real_outputs,
                                        discriminator_fake_outputs, gamma):
    """Return the non-saturating GAN losses for the generator and the
    discriminator.

    (Generator)      -log(sigmoid(D(G(z))))
    (Discriminator)  -gamma * log(sigmoid(D(x))) - log(1 - sigmoid(D(G(z))))
    """
    discriminator_loss_real = gamma * tf.losses.sigmoid_cross_entropy(
        tf.ones_like(discriminator_real_outputs), discriminator_real_outputs)
    discriminator_loss_fake = tf.losses.sigmoid_cross_entropy(
        tf.zeros_like(discriminator_fake_outputs), discriminator_fake_outputs)
    discriminator_loss = discriminator_loss_real + discriminator_loss_fake
    generator_loss = tf.losses.sigmoid_cross_entropy(
        tf.ones_like(discriminator_fake_outputs), discriminator_fake_outputs)
    return generator_loss, discriminator_loss

def wasserstein_gan_losses(discriminator_real_outputs,
                           discriminator_fake_outputs):
    """Return the Wasserstein GAN losses for the generator and the
    discriminator.

    (Generator)      -D(G(z))
    (Discriminator)  -D(x) + D(G(z))
    """
    generator_loss = -tf.reduce_mean(discriminator_fake_outputs)
    discriminator_loss = (
        -tf.reduce_mean(discriminator_real_outputs) - generator_loss)
    return generator_loss, discriminator_loss

def imbalanced_wasserstein_gan_losses(discriminator_real_outputs,
                                      discriminator_fake_outputs, gamma):
    """Return the Wasserstein GAN losses for the generator and the
    discriminator.

    (Generator)      -D(G(z))
    (Discriminator)  -gamma * D(x) + D(G(z))
    """
    generator_loss = -tf.reduce_mean(discriminator_fake_outputs)
    discriminator_loss = (
        -gamma * tf.reduce_mean(discriminator_real_outputs) - generator_loss)
    return generator_loss, discriminator_loss

def hinge_gan_losses(discriminator_real_outputs, discriminator_fake_outputs):
    """Return the hinge GAN losses for the generator and the discriminator.

    (Generator)      -D(G(z))
    (Discriminator)  max(0, 1 - D(x)) + max(0, 1 + D(G(z)))
    """
    generator_loss = -tf.reduce_mean(discriminator_fake_outputs)
    discriminator_loss = (
        tf.reduce_mean(tf.nn.relu(1. - discriminator_real_outputs))
        + tf.reduce_mean(tf.nn.relu(1. + discriminator_fake_outputs)))
    return generator_loss, discriminator_loss

def imbalanced_hinge_gan_losses(discriminator_real_outputs,
                                discriminator_fake_outputs, gamma):
    """Return the hinge GAN losses for the generator and the discriminator.

    (Generator)      -D(G(z))
    (Discriminator)  gamma * max(0, 1 - D(x)) + max(0, 1 + D(G(z)))
    """
    generator_loss = -tf.reduce_mean(discriminator_fake_outputs)
    discriminator_loss = (
        gamma * tf.reduce_mean(tf.nn.relu(1. - discriminator_real_outputs))
        + tf.reduce_mean(tf.nn.relu(1. + discriminator_fake_outputs)))
    return generator_loss, discriminator_loss

def least_squares_gan_losses(discriminator_real_outputs,
                             discriminator_fake_outputs):
    """Return the least-squares GAN losses for the generator and the
    discriminator.

    (Generator)      1/2 * (D(G(z)) - 1 ) ^ 2
    (Discriminator)  1/2 * ((D(x) - 1) ^ 2 + D(G(z)) ^ 2)
    """
    generator_loss = 0.5 * tf.reduce_mean(
        tf.squared_difference(discriminator_fake_outputs, 1.))
    discriminator_loss = 0.5 * (
        tf.reduce_mean(tf.squared_difference(discriminator_real_outputs, 1.)) +
        tf.reduce_mean(tf.square(discriminator_fake_outputs)))
    return generator_loss, discriminator_loss

def absolute_gan_losses(discriminator_real_outputs, discriminator_fake_outputs):
    """Return the absolute GAN losses for the generator and the discriminator.

    (Generator)      -D(G(z))
    (Discriminator)  |D(x)| + D(G(z))
    """
    generator_loss = -tf.reduce_mean(discriminator_fake_outputs)
    discriminator_loss = (
        tf.reduce_mean(tf.abs(discriminator_real_outputs)) - generator_loss)
    return generator_loss, discriminator_loss

def double_absolute_gan_losses(discriminator_real_outputs,
                               discriminator_fake_outputs):
    """Return the double absolute GAN losses for the generator and the
    discriminator.

    (Generator)      |1 - D(G(z))|
    (Discriminator)  |1 - D(x)| + |D(G(z))|
    """
    discriminator_loss = (
        tf.reduce_mean(tf.abs(1. - discriminator_real_outputs))
        + tf.reduce_mean(tf.abs(discriminator_fake_outputs)))
    generator_loss = tf.reduce_mean(tf.abs(1. - discriminator_fake_outputs))
    return generator_loss, discriminator_loss

def new_gan_losses(discriminator_real_outputs, discriminator_fake_outputs):
    """Return the new GAN losses for the generator and the discriminator.

    (Generator)      -D(G(z))
    (Discriminator)  -log(sigmoid(D(x))) - log(1 - sigmoid(D(G(z))))
    """
    discriminator_loss_real = tf.losses.sigmoid_cross_entropy(
        tf.ones_like(discriminator_real_outputs), discriminator_real_outputs)
    discriminator_loss_fake = tf.losses.sigmoid_cross_entropy(
        tf.zeros_like(discriminator_fake_outputs), discriminator_fake_outputs)
    discriminator_loss = discriminator_loss_real + discriminator_loss_fake
    generator_loss = -tf.reduce_mean(discriminator_fake_outputs)
    return generator_loss, discriminator_loss

def relativistic_average_gan_losses(discriminator_real_outputs,
                                    discriminator_fake_outputs):
    """Return the relativistic GAN losses for the generator and the
    discriminator.

    (Generator)      -D(G(z))
    (Discriminator)  -log(sigmoid(D(x))) - log(1 - sigmoid(D(G(z))))
    """
    avg_dis_output_real = tf.reduce_mean(discriminator_real_outputs)
    avg_dis_output_fake = tf.reduce_mean(discriminator_fake_outputs)
    rel_dis_real_outputs = tf.sigmoid(
        discriminator_real_outputs - avg_dis_output_fake)
    rel_dis_fake_outputs = tf.sigmoid(
        discriminator_fake_outputs - avg_dis_output_real)
    discriminator_loss = -tf.reduce_mean(
        tf.log(rel_dis_real_outputs) + tf.log(1. - rel_dis_fake_outputs))
    generator_loss = -tf.reduce_mean(
        tf.log(rel_dis_fake_outputs) + tf.log(1. - rel_dis_real_outputs))
    return generator_loss, discriminator_loss

def relativistic_average_hinge_gan_losses(discriminator_real_outputs,
                                          discriminator_fake_outputs):
    """Return the relativistic GAN losses for the generator and the
    discriminator.

    (Generator)      -D(G(z))
    (Discriminator)  -log(sigmoid(D(x))) - log(1 - sigmoid(D(G(z))))
    """
    avg_dis_output_real = tf.reduce_mean(discriminator_real_outputs)
    avg_dis_output_fake = tf.reduce_mean(discriminator_fake_outputs)
    rel_dis_real_outputs = discriminator_real_outputs - avg_dis_output_fake
    rel_dis_fake_outputs = discriminator_fake_outputs - avg_dis_output_real
    discriminator_loss = (
        tf.reduce_mean(tf.nn.relu(1. - rel_dis_real_outputs)) +
        tf.reduce_mean(tf.nn.relu(1. + rel_dis_fake_outputs)))
    generator_loss = (
        tf.reduce_mean(tf.nn.relu(1. - rel_dis_fake_outputs)) +
        tf.reduce_mean(tf.nn.relu(1. + rel_dis_real_outputs)))
    return generator_loss, discriminator_loss

def improved_classic_gan_losses(discriminator_real_outputs,
                                discriminator_fake_outputs):
    """Return the improved classic GAN losses for the generator and the
    discriminator.

    (Generator)      max(0, -D(G(z)))
    (Discriminator)  -log(sigmoid(D(x))) - log(1 - sigmoid(D(G(z))))
    """
    discriminator_loss_real = tf.losses.sigmoid_cross_entropy(
        tf.ones_like(discriminator_real_outputs), discriminator_real_outputs)
    discriminator_loss_fake = tf.losses.sigmoid_cross_entropy(
        tf.zeros_like(discriminator_fake_outputs), discriminator_fake_outputs)
    discriminator_loss = discriminator_loss_real + discriminator_loss_fake
    generator_loss = tf.reduce_mean(tf.nn.relu(-discriminator_fake_outputs))
    return generator_loss, discriminator_loss

def improved_hinge_gan_losses(discriminator_real_outputs,
                              discriminator_fake_outputs):
    """Return the improved hinge GAN losses for the generator and the
    discriminator.

    (Generator)      max(0, 1 - D(G(z)))
    (Discriminator)  max(0, 1 - D(x)) + max(0, 1 + D(G(z)))
    """
    generator_loss = tf.reduce_mean(tf.nn.relu(1. - discriminator_fake_outputs))
    discriminator_loss = (
        tf.reduce_mean(tf.nn.relu(1. - discriminator_real_outputs))
        + tf.reduce_mean(tf.nn.relu(1. + discriminator_fake_outputs)))
    return generator_loss, discriminator_loss

def improved_hinge_alternative_gan_losses(discriminator_real_outputs,
                                          discriminator_fake_outputs):
    """Return the improved hinge GAN losses for the generator and the
    discriminator.

    (Generator)      max(0, -D(G(z)))
    (Discriminator)  max(0, 1 - D(x)) + max(0, 1 + D(G(z)))
    """
    generator_loss = tf.reduce_mean(tf.nn.relu(-discriminator_fake_outputs))
    discriminator_loss = (
        tf.reduce_mean(tf.nn.relu(1. - discriminator_real_outputs))
        + tf.reduce_mean(tf.nn.relu(1. + discriminator_fake_outputs)))
    return generator_loss, discriminator_loss

def nonsaturating_hinge_gan_losses(discriminator_real_outputs,
                                   discriminator_fake_outputs):
    """Return the nonsaturating hinge GAN losses for the generator and the
    discriminator.

    (Generator)      -log(sigmoid(D(G(z))))
    (Discriminator)  max(0, 1 - D(x)) + max(0, 1 + D(G(z)))
    """
    generator_loss = tf.losses.sigmoid_cross_entropy(
        tf.ones_like(discriminator_fake_outputs), discriminator_fake_outputs)
    discriminator_loss = (
        tf.reduce_mean(tf.nn.relu(1. - discriminator_real_outputs))
        + tf.reduce_mean(tf.nn.relu(1. + discriminator_fake_outputs)))
    return generator_loss, discriminator_loss

def minimax_hinge_gan_losses(discriminator_real_outputs,
                             discriminator_fake_outputs):
    """Return the minimax hinge GAN losses for the generator and the
    discriminator.

    (Generator)      -max(0, 1 + D(G(z)))
    (Discriminator)  max(0, 1 - D(x)) + max(0, 1 + D(G(z)))
    """
    discriminator_loss = (
        tf.reduce_mean(tf.nn.relu(1. - discriminator_real_outputs))
        + tf.reduce_mean(tf.nn.relu(1. + discriminator_fake_outputs)))
    generator_loss = -discriminator_loss
    return generator_loss, discriminator_loss
