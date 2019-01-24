"""This script trains a model."""
import os
import logging
import argparse
from pprint import pformat
import numpy as np
import tensorflow as tf
from dan.config import LOGLEVEL, LOG_FORMAT
from dan.data import load_data, get_dataset
from dan.model import Model
from dan.utils import make_sure_path_exists, load_yaml
from dan.utils import backup_src, update_not_none, setup_loggers
LOGGER = logging.getLogger("dan.train")

def parse_arguments():
    """Parse and return the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', help="Directory to save all the results.")
    parser.add_argument('--config', help="Path to the configuration file.")
    parser.add_argument('--gpu', '--gpu_device_num', type=str, default="0",
                        help="The GPU device number to use.")
    parser.add_argument('--n_jobs', type=int,
                        help="Number of parallel calls to use for input "
                             "pipeline. Set to 1 to disable multiprocessing.")
    args = parser.parse_args()
    return args

def setup_dirs(config):
    """Setup an experiment directory structure and update the `config`
    dictionary with the directory paths."""
    # Get experiment directory structure
    config['exp_dir'] = os.path.realpath(config['exp_dir'])
    config['src_dir'] = os.path.join(config['exp_dir'], 'src')
    config['model_dir'] = os.path.join(config['exp_dir'], 'model')
    config['log_dir'] = os.path.join(config['exp_dir'], 'logs', 'train')

    # Make sure directories exist
    for key in ('log_dir', 'model_dir', 'src_dir'):
        make_sure_path_exists(config[key])

def setup():
    """Parse command line arguments, load configurations, setup environment and
    setup loggers."""
    # Parse the command line arguments
    args = parse_arguments()

    # Load training configurations
    config = load_yaml(args.config)
    update_not_none(config, vars(args))

    # Setup experiment directories and update them to configurations
    setup_dirs(config)

    # Setup loggers
    del logging.getLogger('tensorflow').handlers[0]
    setup_loggers(config['log_dir'])

    # Setup GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']

    # Backup source code
    backup_src(config['src_dir'])

    return config

def load_training_data(config):
    """Load and return the training data."""
    # Load data
    LOGGER.info("Loading training data.")
    train_x = load_data(config['data_source'], config['train_x_filename'])
    train_y = load_data(config['data_source'], config['train_y_filename'])
    val_x = load_data(config['data_source'], config['val_x_filename'])
    val_y = load_data(config['data_source'], config['val_y_filename'])
    LOGGER.info("Training data size: %d", len(train_x))
    LOGGER.info("Validation data size: %d", len(val_x))

    # Build datasets and create iterators
    LOGGER.info("Building dataset.")
    train_dataset = get_dataset(
        train_x, train_y, config['batch_size'], config['data_shape'],
        config['n_classes'], True)
    val_dataset = get_dataset(
        val_x, val_y, config['batch_size'], config['data_shape'],
        config['n_classes'])

    return train_dataset, val_dataset, len(val_x)

def main():
    """Main function."""
    # Setup
    logging.basicConfig(level=LOGLEVEL, format=LOG_FORMAT)
    config = setup()
    LOGGER.info("Using configurations:\n%s", pformat(config))

    # ================================== Data ==================================
    # Load training data
    train_dataset, val_dataset, val_size = load_training_data(config)
    train_x, train_y = train_dataset.make_one_shot_iterator().get_next()
    val_iter = val_dataset.make_initializable_iterator()
    val_x, val_y = val_iter.get_next()
    val_iter_init_op = val_iter.initializer

    # ================================= Model ==================================
    # Build model
    model = Model(config)
    train_nodes = model(x=train_x, y=train_y, mode='train', config=config)
    val_nodes = model(x=val_x, y=val_y, mode='predict', config=config)

    # Log number of parameters in the model
    def get_n_params(var_list):
        """Return the number of variables in a variable list."""
        return int(np.sum([np.product(
            [x.value for x in var.get_shape()]) for var in var_list]))

    LOGGER.info("Number of trainable parameters in {}: {:,}".format(
        model.name, get_n_params(tf.trainable_variables(model.name))))
    for component in model.components:
        LOGGER.info("Number of trainable parameters in {}: {:,}".format(
            component.name, get_n_params(tf.trainable_variables(
                model.name + '/' + component.name))))

    # ========================== Training Preparation ==========================
    # Get tensorflow session config
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # Training hooks
    global_step = tf.train.get_global_step()
    steps_per_iter = (
        config['n_gen_steps_per_iter'] + config['n_dis_steps_per_iter'])
    hooks = [tf.train.NanTensorHook(train_nodes['loss'])]

    # Tensor logger
    tensor_logger = {
        'step': train_nodes['gen_step'],
        'gen_loss': train_nodes['gen_loss'],
        'dis_loss': train_nodes['dis_loss'],
        'train_acc': train_nodes['accuracy']
    }
    train_logger = open(
        os.path.join(config['log_dir'], 'train_losses.log'), 'w')
    val_logger = open(os.path.join(config['log_dir'], 'val_acc.log'), 'w')

    # ======================= Monitored Training Session =======================
    LOGGER.info("Training start.")
    with tf.train.MonitoredTrainingSession(
        save_checkpoint_steps=config['save_checkpoint_steps'] * steps_per_iter,
        checkpoint_dir=config['model_dir'], log_step_count_steps=0,
        hooks=hooks, config=tf_config) as sess:

        # Get global step value
        step = tf.train.global_step(sess, global_step)
        if step == 0:
            train_logger.write("# step, gen_loss, dis_loss, train_acc\n")
            val_logger.write("# step, val_acc\n")

        # ============================== Training ==============================
        if step >= config['steps']:
            LOGGER.info("Global step has already exceeded total steps.")
            train_logger.close()
            val_logger.close()
            return

        # Training iteration
        while step < config['steps']:

            # Train the discriminator
            for _ in range(config['n_dis_steps_per_iter']):
                sess.run(train_nodes['train_ops']['dis'])

            # Train the generator
            log_loss_steps = config['log_loss_steps'] or 100
            if (step + 1) % log_loss_steps == 0:
                for _ in range(config['n_gen_steps_per_iter'] - 1):
                    sess.run(train_nodes['train_ops']['gen'])

                step, _, tensor_logger_values = sess.run([
                    train_nodes['gen_step'], train_nodes['train_ops']['gen'],
                    tensor_logger])
                # Logger
                if config['log_loss_steps'] > 0:
                    LOGGER.info("step={}, {}".format(step, ', '.join([
                        '{}={: 8.4E}'.format(key, value)
                        for key, value in tensor_logger_values.items()
                        if key != 'step'])))
                train_logger.write("{}, {: 10.6E}, {: 10.6E}, {: .4E}\n".format(
                    step, tensor_logger_values['gen_loss'],
                    tensor_logger_values['dis_loss'],
                    tensor_logger_values['train_acc']))
            else:
                for _ in range(config['n_gen_steps_per_iter'] - 1):
                    sess.run([
                        train_nodes['gen_step'],
                        train_nodes['train_ops']['gen']])
                step, _ = sess.run([
                    train_nodes['gen_step'], train_nodes['train_ops']['gen']])

            # Run validation
            if ((config['validation_steps'] > 0)
                    and (step % config['validation_steps'] == 0)):
                LOGGER.info("Running validation")
                sess.run(val_iter_init_op)
                count = 0
                for _ in range((val_size // config['batch_size'])):
                    count += sess.run(val_nodes['correct_predictions_counts'])
                val_acc = count / val_size
                if config['log_loss_steps'] > 0:
                    LOGGER.info("step={}, val_acc={:.4}".format(step, val_acc))
                val_logger.write("{}, {:.6}\n".format(step, val_acc))

            # Stop training if stopping criterion suggests
            if sess.should_stop():
                break

    LOGGER.info("Training end.")
    train_logger.close()
    val_logger.close()

if __name__ == "__main__":
    main()
