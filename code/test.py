from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import sys
import tables
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.python.ops import control_flow_ops
from nets import nets_factory
from auxiliary import losses
from roc_curve import calculate_roc
import matplotlib.pyplot as plt

slim = tf.contrib.slim

######################
# Train Directory #
######################

tf.app.flags.DEFINE_string(
    'train_dir', 'TRAIN_CNN_3D/train_logs',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 1,
    'The frequency with which logs are print.')


######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'adam',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')


tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 5.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'model_speech_name', 'lipread_speech', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'model_mouth_name', 'lipread_mouth', 'The name of the architecture to train.')


tf.app.flags.DEFINE_integer(
    'batch_size', 128, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'num_epochs', 20, 'The number of epochs for training.')


#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune. ex:/home/sina/TRAIN_CASIA/train_logs/vgg_19.cpkt')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring'
    'from a checkpoint. ex: vgg_19/fc8/biases,vgg_19/fc8/weights')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

# Store all elemnts in FLAG structure!
FLAGS = tf.app.flags.FLAGS


def _configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.

    Returns:
      A `Tensor` representing the learning rate.

    Raises:
      ValueError: if
    """
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)
    if FLAGS.sync_replicas:
        decay_steps /= FLAGS.replicas_to_aggregate

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.

    Args:
      learning_rate: A scalar or `Tensor` learning rate.

    Returns:
      An instance of an optimizer.

    Raises:
      ValueError: if FLAGS.optimizer is not recognized.
    """

    if FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def _get_variables_to_train():
    """Returns a list of variables to train.

    Returns:
      A list of variables to train by the optimizer.
    """
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


# Definign arbitrary data
num_training_samples = 1000
num_testing_samples = 1000
train_data = {}

train_data = {'mouth': np.random.random_sample(size=(num_training_samples, 9, 60, 100, 1)),
              'speech': np.random.random_sample(size=(num_training_samples, 15, 40, 1, 3))}
test_data = {'mouth': np.random.random_sample(size=(num_testing_samples, 9, 60, 100, 1)),
             'speech': np.random.random_sample(size=(num_testing_samples, 15, 40, 1, 3))}

train_label = np.random.randint(2, size=(num_training_samples, 1))
test_label = np.random.randint(2, size=(num_testing_samples, 1))


# # Uncomment if data standardalization is required and the mean and std vectors have been calculated.
# ############ Get the mean vectors ####################
#
# # mean mouth
# mean_mouth = np.load('/path/to/mean/file/mouth.npy')
# # mean_mouth = np.tile(mean_mouth.reshape(47, 73, 1), (1, 1, 9))
# mean_mouth = mean_mouth[None, :]
# mean_channel_mouth = np.mean(mean_mouth)
#
# # mean speech
# mean_speech = np.load('/path/to/mean/file/speech.npy')
# mean_speech = mean_speech[None, :]
# # mean_channel_speech = np.hstack((
# #     [np.mean(mean_speech[:, :, :, 0])], [np.mean(mean_speech[:, :, :, 1])], [np.mean(mean_speech[:, :, :, 2])]))
#
# ############ Get the std vectors ####################
#
# # mean std
# std_mouth = np.load('/path/to/std/file/mouth.npy')
# std_mouth = np.tile(std_mouth.reshape(60, 100, 1), (1, 1, 9))
# std_mouth = std_mouth[None, :]
#
# # mean speech
# std_speech = np.load('/path/to/std/file/speech.npy')
# std_speech = std_speech[None, :]




def main(_):


    tf.logging.set_verbosity(tf.logging.INFO)

    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):
        ######################
        # Config model_deploy#
        ######################

        # required from data
        num_samples_per_epoch = train_data['mouth'].shape[0]
        num_batches_per_epoch = int(num_samples_per_epoch / FLAGS.batch_size)

        num_samples_per_epoch_test = test_data['mouth'].shape[0]
        num_batches_per_epoch_test = int(num_samples_per_epoch_test / FLAGS.batch_size)

        # Create global_step
        global_step = tf.Variable(0, name='global_step', trainable=False)

        #########################################
        # Configure the larning rate. #
        #########################################
        learning_rate = _configure_learning_rate(num_samples_per_epoch, global_step)
        opt = _configure_optimizer(learning_rate)

        ######################
        # Select the network #
        ######################
        is_training = tf.placeholder(tf.bool)

        network_speech_fn = nets_factory.get_network_fn(
            FLAGS.model_speech_name,
            num_classes=2,
            weight_decay=FLAGS.weight_decay,
            is_training=is_training)

        network_mouth_fn = nets_factory.get_network_fn(
            FLAGS.model_mouth_name,
            num_classes=2,
            weight_decay=FLAGS.weight_decay,
            is_training=is_training)

        #####################################
        # Select the preprocessing function #
        #####################################

        # TODO: Do some preprocessing if necessary.

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        # with tf.device(deploy_config.inputs_device()):
        """
        Define the place holders and creating the batch tensor.
        """

        # Mouth spatial set
        INPUT_SEQ_LENGTH = 9
        INPUT_HEIGHT = 60
        INPUT_WIDTH = 100
        INPUT_CHANNELS = 1
        batch_mouth = tf.placeholder(tf.float32, shape=(
            [None, INPUT_SEQ_LENGTH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS]))

        # Speech spatial set
        INPUT_SEQ_LENGTH_SPEECH = 15
        INPUT_HEIGHT_SPEECH = 40
        INPUT_WIDTH_SPEECH = 1
        INPUT_CHANNELS_SPEECH = 3
        batch_speech = tf.placeholder(tf.float32, shape=(
            [None, INPUT_SEQ_LENGTH_SPEECH, INPUT_HEIGHT_SPEECH, INPUT_WIDTH_SPEECH, INPUT_CHANNELS_SPEECH]))

        # Label
        batch_labels = tf.placeholder(tf.uint8, (None, 1))
        margin_imp_tensor = tf.placeholder(tf.float32, ())

        ################################
        ## Feed forwarding to network ##
        ################################
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_clones):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                        """
                        Two distance metric are defined:
                           1 - distance_weighted: which is a weighted average of the distance between two structures.
                           2 - distance_l2: which is the regular l2-norm of the two networks outputs.
                        Place holders

                        """
                        ########################################
                        ######## Outputs of two networks #######
                        ########################################

                        logits_speech, end_points_speech = network_speech_fn(batch_speech)
                        logits_mouth, end_points_mouth = network_mouth_fn(batch_mouth)

                        # # Uncomment if the output embedding is desired to be as |f(x)| = 1
                        # logits_speech = tf.nn.l2_normalize(logits_speech, dim=1, epsilon=1e-12, name=None)
                        # logits_mouth = tf.nn.l2_normalize(logits_mouth, dim=1, epsilon=1e-12, name=None)

                        #################################################
                        ########### Loss Calculation ####################
                        #################################################

                        # ##### Weighted distance using a fully connected layer #####
                        # distance_vector = tf.subtract(logits_speech, logits_mouth,  name=None)
                        # distance_weighted = slim.fully_connected(distance_vector, 1, activation_fn=tf.nn.sigmoid,
                        #                                          normalizer_fn=None,
                        #                                          scope='fc_weighted')

                        ##### Euclidean distance ####
                        distance_l2 = tf.sqrt(
                            tf.reduce_sum(tf.pow(tf.subtract(logits_speech, logits_mouth), 2), 1, keep_dims=True))

                        ##### Contrastive loss ######
                        loss = losses.contrastive_loss(batch_labels, distance_l2, margin_imp=margin_imp_tensor,
                                                       scope=scope)

                        # ##### call the optimizer ######
                        # # TODO: call optimizer object outside of this gpu environment
                        #
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)


        # Calculate the mean of each gradient.
        grads = average_gradients(tower_grads)

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Track the moving averages of all trainable variables.
        MOVING_AVERAGE_DECAY = 0.9999
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        #################################################
        ########### Summary Section #####################
        #################################################

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Add summaries for all end_points.
        for end_point in end_points_speech:
            x = end_points_speech[end_point]
            # summaries.add(tf.summary.histogram('activations_speech/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity_speech/' + end_point,
                                            tf.nn.zero_fraction(x)))

        for end_point in end_points_mouth:
            x = end_points_mouth[end_point]
            # summaries.add(tf.summary.histogram('activations_mouth/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity_mouth/' + end_point,
                                            tf.nn.zero_fraction(x)))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        # Add to parameters to summaries
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))
        summaries.add(tf.summary.scalar('eval/Loss', loss))
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

    ###########################
    ######## Training #########
    ###########################

    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        # Initialization of the network.
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore, max_to_keep=20)
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Restore the model
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir='TRAIN_CNN_3D/')
        saver.restore(sess, latest_checkpoint)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=graph)

        ###################################################
        ############################ TEST  ################
        ###################################################
        score_dissimilarity_vector = np.zeros((FLAGS.batch_size * num_batches_per_epoch_test, 1))
        label_vector = np.zeros((FLAGS.batch_size * num_batches_per_epoch_test, 1))

        # Loop over all batches
        for i in range(num_batches_per_epoch_test):
            start_idx = i * FLAGS.batch_size
            end_idx = (i + 1) * FLAGS.batch_size
            speech_test, mouth_test, label_test = test_data['speech'][start_idx:end_idx], test_data['mouth'][
                                                                                          start_idx:end_idx], test_label[
                                                                                                              start_idx:end_idx]

            # # # Uncomment if standardalization is needed
            # # mean subtraction if necessary
            # speech_test = (speech_test - mean_speech) / std_speech
            # mouth_test = (mouth_test - mean_mouth) / std_mouth

            # Evaluation phase
            # WARNING: margin_imp_tensor has no effect here but it needs to be there because its tensor required a value to feed in!!
            loss_value, score_dissimilarity, _ = sess.run([loss, distance_l2, is_training],
                                                          feed_dict={is_training: False,
                                                                     margin_imp_tensor: 50,
                                                                     batch_speech: speech_test,
                                                                     batch_mouth: mouth_test,
                                                                     batch_labels: label_test.reshape(
                                                                         [FLAGS.batch_size, 1])})
            if (i + 1) % FLAGS.log_every_n_steps == 0:
                print("TESTING:" + ", Minibatch " + str(
                    i + 1) + " of %d " % num_batches_per_epoch_test)
            score_dissimilarity_vector[start_idx:end_idx] = score_dissimilarity
            label_vector[start_idx:end_idx] = label_test

        ##############################
        ##### K-fold validation ######
        ##############################
        K = 10
        EER = np.zeros((K, 1))
        AUC = np.zeros((K, 1))
        AP = np.zeros((K, 1))
        batch_k_validation = int(label_vector.shape[0] / float(K))

        for i in range(K):
            EER[i, :], AUC[i, :], AP[i, :], fpr, tpr = calculate_roc.calculate_eer_auc_ap(
                label_vector[i * batch_k_validation:(i + 1) * batch_k_validation],
                score_dissimilarity_vector[i * batch_k_validation:(i + 1) * batch_k_validation])

        # Printing Equal Error Rate(EER), Area Under the Curve(AUC) and Average Precision(AP)
        print("TESTING:" +", EER= " + str(np.mean(EER, axis=0)) + ", AUC= " + str(
            np.mean(AUC, axis=0)) + ", AP= " + str(np.mean(AP, axis=0)))


if __name__ == '__main__':
    tf.app.run()
