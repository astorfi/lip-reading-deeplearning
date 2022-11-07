"""
Contrastive cost
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_slim import arg_scope
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.util.deprecation import deprecated

import tensorflow
tf = tensorflow.compat.v1

# def contrastive_loss(onehot_labels, logits, margin=1, scope=None):
#     """With this definition the loss will be calculated.
#         Args:
#           y: The labels.
#           distance: The distance vector between the output features..
#           batch_size: the batch size is necessary because the loss calculation would be over each batch.
#         Returns:
#           The total loss.
#     """
#     with ops.name_scope(scope, "contrastive_loss", [onehot_labels, logits]) as scope:
#         # logits.get_shape().assert_is_compatible_with(onehot_labels.get_shape())
#
#         onehot_labels = math_ops.cast(onehot_labels, logits.dtype)
#
#         term_1 = tf.multiply(onehot_labels, tf.square(logits))[:,0:1]
#         term_2 = tf.multiply(onehot_labels, tf.square(tf.maximum((margin - logits), 0)))[:,1:]
#
#         # Contrastive
#         Contrastive_Loss = tf.add(term_1, term_2) / 2
#         loss = tf.losses.compute_weighted_loss(Contrastive_Loss, scope=scope)
#
#         return tf.losses.compute_weighted_loss(Contrastive_Loss, scope=scope)

def contrastive_loss(labels, logits, margin_gen=0, margin_imp=1, scope=None):
    """With this definition the loss will be calculated.
        Args:
          y: The labels.
          distance: The distance vector between the output features..
          batch_size: the batch size is necessary because the loss calculation would be over each batch.
        Returns:
          The total loss.
    """
    with ops.name_scope(scope, "contrastive_loss", [labels, logits]) as scope:
        # logits.get_shape().assert_is_compatible_with(onehot_labels.get_shape())

        labels = math_ops.cast(labels, logits.dtype)

        # term_1 = tf.multiply(labels, tf.square(logits))
        term_1 = tf.multiply(labels, tf.square(tf.maximum((logits - margin_gen), 0)))
        term_2 = tf.multiply(1 - labels, tf.square(tf.maximum((margin_imp - logits), 0)))

        # Contrastive
        Contrastive_Loss = tf.add(term_1, term_2) / 2
        loss = tf.losses.compute_weighted_loss(Contrastive_Loss, scope=scope)

        return loss


# def contrastive_loss(onehot_labels, logits, batch_size, margin=1):
#     """With this definition the loss will be calculated.
#         Args:
#           y: The labels.
#           distance: The distance vector between the output features..
#           batch_size: the batch size is necessary because the loss calculation would be over each batch.
#         Returns:
#           The total loss.
#     """
#     with ops.name_scope(scope, "contrastive_loss", [onehot_labels, logits]) as scope:
#         logits.get_shape().assert_is_compatible_with(onehot_labels.get_shape())
#
#         onehot_labels = math_ops.cast(onehot_labels, logits.dtype)
#
#         term_1 = tf.multiply(onehot_labels, tf.square(distance))[:,0:1]
#         term_2 = tf.multiply(onehot_labels, tf.square(tf.maximum((margin - distance), 0)))[:,1:]
#
#         # Contrastive
#         Contrastive_Loss = tf.add(term_1, term_2) / batch_size / 2
#         tf.add_to_collection('losses', Contrastive_Loss)
#
#         return tf.add_n(tf.get_collection('losses'), name='total_loss')

