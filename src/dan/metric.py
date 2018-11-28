"""Class for metrics."""
import tensorflow as tf
from collections import OrderedDict

class Metric(object):
    def __init__(self, y, predictions):
        with tf.variable_scope("metrics") as scope:
            self.metric_dict = self.build(y, predictions)

            self.metrics = tf.get_collection('metrics', scope)
            self.updates = tf.group(*tf.get_collection('updates'), scope)
            self.vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope)


    def run_metrics(self, sess):
	pass

class BinaryMetric(Metric):
    def build(self, y, predictions):
        """Build all the metrics."""
        metric_dict = OrderedDict()

        metric_dict['true_postives'] = tf.metrics.true_positives(
            y, predictions, metrics_collections='metrics',
            updates_collections='updates')
        metric_dict['true_negatives'] = tf.metrics.true_negatives(
            y, predictions, metrics_collections='metrics',
            updates_collections='updates')
        metric_dict['false_postives'] = tf.metrics.false_positives(
            y, predictions, metrics_collections='metrics',
            updates_collections='updates')
        metric_dict['false_negatives'] = tf.metrics.false_negatives(
            y, predictions, metrics_collections='metrics',
            updates_collections='updates')
        metric_dict['mean_absolute_error'] = tf.metrics.mean_absolute_error(
            y, predictions, metrics_collections='metrics',
            updates_collections='updates')
        metric_dict['mean_cosine_distance'] = tf.metrics.mean_cosine_distance(
            y, predictions, 1, metrics_collections='metrics',
            updates_collections='updates')
        metric_dict['mean_squared_error'] = tf.metrics.mean_squared_error(
            y, predictions, metrics_collections='metrics',
            updates_collections='updates')
        metric_dict['auc'] = tf.metrics.auc(
            y, predictions, metrics_collections='metrics',
            updates_collections='updates')

        metric_dict['recall'] = (metric_dict['true_postives'] /
            (metric_dict['true_postives'] + metric_dict['false_negatives']))
        metric_dict['precision'] = (metric_dict['true_postives'] /
            (metric_dict['true_postives'] + metric_dict['false_positives']))

        metric_dict['f1'] = (metric_dict['precision'] * metric_dict['recall'] /
            (metric_dict['precision'] + metric_dict['recall'])) * 2.

        return metric_dict

