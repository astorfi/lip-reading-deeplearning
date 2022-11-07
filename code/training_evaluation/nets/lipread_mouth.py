# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

@@vgg_a
@@vgg_16
@@vgg_19
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow
tf = tensorflow.compat.v1

import tf_slim as slim
LSTM_status = False


def lipread_mouth_arg_scope(is_training, weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  # Add normalizer_fn=slim.batch_norm if Batch Normalization is required!
  with slim.arg_scope([slim.conv3d, slim.fully_connected],
                      activation_fn=None,
                      weights_initializer=tf.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG'),
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      normalizer_fn=slim.batch_norm,
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv3d], padding='SAME') as arg_sc:
      return arg_sc

def PReLU(input,scope):
  """
  Similar to TFlearn implementation
  :param input: input of the PReLU which is output of a layer.
  :return: The output.
  """
  alphas = tf.get_variable(scope, input.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)

  return tf.nn.relu(input) + alphas * (input - abs(input)) * 0.5


def mouth_cnn_lstm(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.8,
          spatial_squeeze=True,
          scope='mouth_cnn'):
  """Oxford Net VGG 11-Layers version A Example.

  Note: All the fully_connected layers have been transformed to conv3d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.

  Returns:
    the last op containing the log predictions and end_points dict.
  """

  end_points = {}
  with tf.variable_scope(scope, 'mouth_cnn', [inputs]) as sc:
    # end_points_collection = sc.name + '_end_points'
    # # Collect outputs for conv3d, fully_connected and max_pool2d.
    # with slim.arg_scope([slim.conv3d, slim.max_pool2d],
    #                     outputs_collections=end_points_collection):

    ##### Convolution Section #####
    # Tensor("batch:1", shape=(?, 9, 60, 100, 1), dtype=float32, device=/device:CPU:0)
    inputs = tf.to_float(inputs)
    net = slim.repeat(inputs, 1, slim.conv3d, 16, [1, 3, 3], scope='conv1')
    net = PReLU(net, 'conv1_activation')
    net = tf.nn.max_pool3d(net, strides=[1, 1, 2, 2, 1], ksize=[1, 1, 3, 3, 1],padding='VALID', name='pool1')
    # net = slim.max_pool2d(net, [3, 3], scope='pool1')
    net = slim.repeat(net, 1, slim.conv3d, 32, [1, 3, 3], scope='conv2')
    net = PReLU(net, 'conv2_activation')
    net = tf.nn.max_pool3d(net, strides=[1, 1, 2, 2, 1], ksize=[1, 1, 3, 3, 1], padding='VALID', name='pool2')

    net = slim.conv3d(net, 64, [1, 3, 3], scope='conv31')
    net = PReLU(net, 'conv31_activation')
    net = slim.conv3d(net, 64, [1, 3, 3], scope='conv32')
    net = PReLU(net, 'conv32_activation')

    net = tf.nn.max_pool3d(net, strides=[1, 1, 2, 2, 1], ksize=[1, 1, 3, 3, 1], padding='VALID', name='pool3')
    net = slim.repeat(net, 1, slim.conv3d, 128, [1, 3, 3], scope='conv4')
    net = PReLU(net, 'conv4_activation')
    net = tf.nn.max_pool3d(net, strides=[1, 1, 2, 2, 1], ksize=[1, 1, 3, 3, 1], padding='VALID', name='pool4')

    ##### FC section #####
    # Use conv3d instead of fully_connected layers.
    net = slim.repeat(net, 1, slim.conv3d, 256, [1, 2, 5], padding='VALID', scope='fc5')
    net = PReLU(net, 'fc5_activation')
    # net = PReLU(net)
    # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
    #                    scope='dropout5')


    if LSTM_status:

      net = slim.conv3d(net, 64, [1, 1, 1], padding='VALID', activation_fn=None, normalizer_fn=None, scope='fc6')
      net = PReLU(net, 'fc6_activation')

      # Tensor("tower_0/speech_cnn/fc6/squeezed:0", shape=(?, 9, 128), dtype=float32, device=/device:GPU:0)
      net = tf.squeeze(net, [2, 3], name='fc6/squeezed')

    else:
      net = slim.conv3d(net, 64, [9, 1, 1],padding='VALID', activation_fn=None, normalizer_fn=None, scope='fc5')

      # Tensor("tower_0/speech_cnn/fc6/squeezed:0", shape=(?, 9, 128), dtype=float32, device=/device:GPU:0)
      net = tf.squeeze(net, [1, 2, 3], name='fc6/squeezed')

    if LSTM_status:
      ##### LSTM-1 #####
      # use sequence_length=X_lengths argument in tf.nn.dynamic_rnn if necessary.
      cell_1 = tf.contrib.rnn.core_rnn_cell.LSTMCell(num_units=128, state_is_tuple=True)
      outputs, last_states = tf.nn.dynamic_rnn(
        cell=cell_1,
        dtype=tf.float32,
        inputs=net,
        scope='LSTM-mouth')
      net = last_states.h

    return net, end_points

