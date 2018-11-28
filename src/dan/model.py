"""This file defines the model."""
import logging
import numpy as np
import tensorflow as tf
from dan.losses import get_adv_losses
from dan.utils import load_component
LOGGER = logging.getLogger(__name__)

class Model:
    """Class that defines the model."""
    def __init__(self, config, name='Model'):
        self.name = name

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:

            # Save the variable scope object
            self.scope = scope

            # Build the model graph
            LOGGER.info("Building model.")
            self.gen = load_component(
                'generator', config['nets']['generator'], 'Generator')(
                    config['n_classes'])
            self.dis = load_component(
                'discriminator', config['nets']['discriminator'],
                'Discriminator')()

            # Save components to a list for showing statistics
            self.components = [self.gen, self.dis]

    def __call__(self, x, y=None, mode=None, params=None, config=None):
        if mode == 'train':
            if y is None:
                raise TypeError("`y` must not be None for 'train' mode.")
            return self.get_train_nodes(x, y, config)
        elif mode == 'predict':
            return self.get_predict_nodes(x, y, config)
        raise ValueError("Unrecognized mode received. Expect 'train' or "
                         "'predict' but get {}".format(mode))

    def get_train_nodes(self, x, y, config):
        """Return a dictionary of graph nodes for training."""
        LOGGER.info("Building training nodes.")
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:

            nodes = {}

            # Get or create global step
            global_step = tf.train.get_or_create_global_step()
            nodes['gen_step'] = tf.get_variable(
                'gen_step', [], tf.int32, tf.constant_initializer(0),
                trainable=False)

            # --- One-hot encoded labels ---------------------------------------
            nodes['real_y'] = tf.reshape(
                tf.one_hot(y, config['n_classes']), (-1, config['n_classes']))

            # --- Generator output ---------------------------------------------
            nodes['fake_y'] = self.gen(x, training=True)

            # --- Discriminator output -----------------------------------------
            nodes['dis_real'] = self.dis(x, nodes['real_y'], True)
            nodes['dis_fake'] = self.dis(x, nodes['fake_y'], True)

            # --- Predictions and accuracy -------------------------------------
            nodes['predictions'] = tf.expand_dims(
                tf.argmax(nodes['fake_y'], 1, output_type=tf.int32), -1)
            nodes['correct_predictions_counts'] = tf.count_nonzero(tf.equal(
                y, nodes['predictions']), dtype=tf.int32)
            nodes['accuracy'] = tf.divide(
                nodes['correct_predictions_counts'], tf.shape(x)[0])

            # ============================= Losses =============================
            LOGGER.info("Building losses.")
            # --- Adversarial losses -------------------------------------------
            nodes['gen_loss'], nodes['dis_loss'] = get_adv_losses(
                nodes['dis_real'], nodes['dis_fake'], config['gan_loss_type'])

            # --- Gradient penalties -------------------------------------------
            if config['use_gradient_penalties']:
                # Compute gradient penalties
                eps_y = tf.random_uniform(tf.shape(nodes['real_y']))
                inter_y = (
                    eps_y * nodes['real_y'] + (1.0 - eps_y) * nodes['fake_y'])
                dis_y_inter_out = self.dis(x, inter_y, True)
                gradient_y = tf.gradients(dis_y_inter_out, inter_y)[0]
                slopes_y = tf.sqrt(1e-8 + tf.reduce_sum(
                    tf.square(gradient_y),
                    np.arange(1, gradient_y.get_shape().ndims)))
                gradient_penalty_y = tf.reduce_mean(tf.square(slopes_y - 1.0))
                nodes['dis_loss'] += 10.0 * gradient_penalty_y

            # Compute total loss (for logging and detecting NAN values only)
            nodes['loss'] = nodes['gen_loss'] + nodes['dis_loss']

            # ========================== Training ops ==========================
            LOGGER.info("Building training ops.")
            # --- Optimizers ---------------------------------------------------
            gen_opt = tf.train.AdamOptimizer(
                config['learning_rate'], config['adam']['beta1'],
                config['adam']['beta2'])
            dis_opt = tf.train.AdamOptimizer(
                config['learning_rate'], config['adam']['beta1'],
                config['adam']['beta2'])

            # --- Training ops -------------------------------------------------
            nodes['train_ops'] = {}
            # Training op for the discriminator
            dis_vas = tf.trainable_variables(scope.name + '/' + self.dis.name)
            nodes['train_ops']['dis'] = dis_opt.minimize(
                nodes['dis_loss'], global_step, dis_vas)

            # Apply weight clipping
            if (config['gan_loss_type'] == 'wasserstein' and
                    not config['use_gradient_penalties']):
                with tf.control_dependencies([nodes['train_ops']['dis']]):
                    nodes['train_ops']['dis'] = tf.group(
                        *(tf.assign(var, tf.clip_by_value(
                            var, -config['weight_clipping_threshold'],
                            config['weight_clipping_threshold']))
                          for var in dis_vas))

            # Training ops for the generator
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            gen_step_increment = tf.assign_add(nodes['gen_step'], 1)
            with tf.control_dependencies(update_ops + [gen_step_increment]):
                nodes['train_ops']['gen'] = gen_opt.minimize(
                    nodes['gen_loss'], global_step,
                    tf.trainable_variables(scope.name + '/' + self.gen.name))

            # =========================== Summaries ============================
            LOGGER.info("Building summaries.")
            if config['save_summaries_steps'] > 0:
                with tf.name_scope('losses'):
                    tf.summary.scalar('gen_loss', nodes['gen_loss'])
                    tf.summary.scalar('dis_loss', nodes['dis_loss'])

        return nodes

    def get_predict_nodes(self, x, y=None, config=None):
        """Return a dictionary of graph nodes for training."""
        LOGGER.info("Building prediction nodes.")
        nodes = {}

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # Generator output an predictions
            nodes['fake_y'] = self.gen(x, training=False)
            nodes['predictions'] = tf.expand_dims(
                tf.argmax(nodes['fake_y'], 1, output_type=tf.int32), -1)

            if y is not None:
                # Compute the accuracy
                nodes['real_y'] = tf.one_hot(y, config['n_classes'])
                nodes['correct_predictions_counts'] = tf.count_nonzero(
                    tf.equal(y, nodes['predictions']), dtype=tf.int32)

        return nodes
