from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import sys
import tables
import numpy as np
from tensorflow.python.ops import control_flow_ops
from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from auxiliary import losses
from preprocessing import preprocessing_factory
from roc_curve import calculate_roc

slim = tf.contrib.slim


######################
# Train Directory #
######################

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', '/home/sina/TRAIN_LIPREAD/train_logs-547',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 2,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 8,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 8,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 20,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 10,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 500,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'sgd',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

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
    'num_epochs_per_decay', 2.0,
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
    'dataset_name', 'lipread', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '/home/sina/datasets/lip_read_features', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_speech_name', 'lipread_speech', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'model_mouth_name', 'lipread_mouth', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 1024 , 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', 125000,
                            'The maximum number of training steps.')

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
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=FLAGS.adadelta_rho,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=FLAGS.ftrl_learning_rate_power,
            initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
            l1_regularization_strength=FLAGS.ftrl_l1,
            l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=FLAGS.momentum,
            name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.rmsprop_momentum,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer


def _add_variables_summaries(learning_rate):
    summaries = []
    for variable in slim.get_model_variables():
        summaries.append(tf.summary.histogram(variable.op.name, variable))
    summaries.append(tf.summary.scalar('training/Learning Rate', learning_rate))
    return summaries


def _get_init_fn():
    """Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """
    if FLAGS.checkpoint_path is None:
        return None

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(FLAGS.train_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % FLAGS.train_dir)
        return None

    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=FLAGS.ignore_missing_vars)


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


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)

    graph = tf.Graph()
    with graph.as_default():
        ######################
        # Config model_deploy#
        ######################
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=FLAGS.task,
            num_replicas=FLAGS.worker_replicas,
            num_ps_tasks=FLAGS.num_ps_tasks)

        # # Create global_step
        # with tf.device(deploy_config.variables_device()):
        #     global_step = slim.create_global_step()

        ###############################
        # Select and load the dataset #
        ###############################

        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        # Load the dataset
        fileh = tables.open_file('/home/sina/datasets/lip_read_features/lipread_mouth_73x47_mfcc_13x?X3_validation.hdf5', mode='r')

        # Get the mean vectors
        mean_mouth = np.load('/home/sina/GITHUB/LIPREAD_PROJECT/data_preprocessing/mean_mouth.npy')
        mean_mouth = mean_mouth.reshape(
            (1, fileh.root.mouth.shape[1], fileh.root.mouth.shape[2], fileh.root.mouth.shape[3]))
        mean_speech = np.load('/home/sina/GITHUB/LIPREAD_PROJECT/data_preprocessing/mean_speech.npy')
        mean_speech = mean_speech.reshape(
            (1, fileh.root.speech.shape[1], fileh.root.speech.shape[2], fileh.root.speech.shape[3]))

        num_samples_per_epoch = fileh.root.label.shape[0]
        num_batches_per_epoch = int(num_samples_per_epoch / FLAGS.batch_size)

        ######################
        # Select the network #
        ######################

        network_speech_fn = nets_factory.get_network_fn(
            FLAGS.model_speech_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            weight_decay=FLAGS.weight_decay,
            is_training=True)

        network_mouth_fn = nets_factory.get_network_fn(
            FLAGS.model_mouth_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            weight_decay=FLAGS.weight_decay,
            is_training=True)

        #####################################
        # Select the preprocessing function #
        #####################################

        #TODO: Do some preprocessing if necessary.

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        with tf.device(deploy_config.inputs_device()):
            """
            Define the place holders and creating the batch tensor.
            """

            # Place holders
            mouth = tf.placeholder(tf.float32, (47, 73, 9))
            speech = tf.placeholder(tf.float32, (13, 15, 1))
            label = tf.placeholder(tf.uint8, (1))

            # Create the batch tensors
            batch_speech, batch_mouth, batch_labels = tf.train.batch(
                [speech, mouth, label],
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size)

        ####################
        # Run the model #
        ####################

        # Outputs of two networks
        logits_speech, end_points_speech = network_speech_fn(batch_speech)
        logits_mouth, end_points_mouth = network_mouth_fn(batch_mouth)

        #############################
        # Specify the loss function #
        #############################

        # Two distance metric are defined:
        #    1 - distance_weighted: which is a weighted average of the distance between two structures.
        #    2 - distance_l2: which is the regular l2-norm of the two networks outputs.

        #### Weighted distance ######
        distance_vector = tf.subtract(logits_speech, logits_mouth, name=None)
        distance_weighted = slim.fully_connected(distance_vector, 1, activation_fn=None, normalizer_fn=None,
                                                 scope='fc_weighted')

        #### Euclidean distance ####
        distance_l2 = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(logits_speech, logits_mouth), 2), 1, keep_dims=True))

        #### Contrastive loss #####
        loss = losses.contrastive_loss(batch_labels, distance_l2, margin=1)

        # Adding the accuracy metric
        with tf.name_scope('accuracy'):
            predictions = tf.to_int64(tf.sign(tf.sign(distance_l2 - 0.5) + 1))
            labels = tf.argmax(distance_l2, 1)
            accuracy = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))
            tf.add_to_collection('accuracy', accuracy)


        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Add summaries for all end_points.
        for end_point in end_points_speech:
            x = end_points_speech[end_point]
            summaries.add(tf.summary.histogram('activations_speech/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity_speech/' + end_point,
                                            tf.nn.zero_fraction(x)))

        for end_point in end_points_mouth:
            x = end_points_mouth[end_point]
            summaries.add(tf.summary.histogram('activations_mouth/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity_mouth/' + end_point,
                                            tf.nn.zero_fraction(x)))

        # # Add summaries for losses.
        # for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
        #     summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        #################################
        # Configure the moving averages #
        #################################
        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

        #########################################
        # Configure the optimization procedure. #
        #########################################
        # deploy_config.optimizer_device()
        # with tf.device(deploy_config.optimizer_device()):
        # learning_rate = _configure_learning_rate(num_samples_per_epoch, global_step)
        # optimizer = _configure_optimizer(learning_rate)
        # optimizer = optimizer.minimize(loss)
        # summaries.add(tf.summary.scalar('learning_rate', learning_rate))
        # if FLAGS.sync_replicas:
        #     # If sync_replicas is enabled, the averaging will be done in the chief
        #     # queue runner.
        #     optimizer = tf.train.SyncReplicasOptimizer(
        #         opt=optimizer,
        #         replicas_to_aggregate=FLAGS.replicas_to_aggregate,
        #         variable_averages=variable_averages,
        #         variables_to_average=moving_average_variables,
        #         replica_id=tf.constant(FLAGS.task, tf.int32, shape=()),
        #         total_num_replicas=FLAGS.worker_replicas)
        # elif FLAGS.moving_average_decay:
        #     # Update ops executed locally by trainer.
        #     update_ops.append(variable_averages.apply(moving_average_variables))
        #
        # summaries.add(tf.summary.scalar('eval/Loss', loss))
        #
        # summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        #
        # # Merge all summaries together.
        # summary_op = tf.summary.merge(list(summaries), name='summary_op')

    ###########################
    # Kicks off the training. #
    ###########################
    with tf.Session(graph=graph) as sess:

        # Initialization of the network.
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(slim.get_variables_to_restore(),max_to_keep=15)
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        num_epoch = 1

        # Save the model
        saver.restore(sess, FLAGS.train_dir)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=graph)

        score_dissimilarity_vector = np.zeros((FLAGS.batch_size * num_batches_per_epoch , 1))
        label_vector = np.zeros((FLAGS.batch_size * num_batches_per_epoch,))


        for epoch in range(num_epoch):

            # Loop over all batches
            for i in range(num_batches_per_epoch):
                start_idx = i * FLAGS.batch_size
                end_idx = (i + 1) * FLAGS.batch_size
                speech, mouth, label = fileh.root.speech[start_idx:end_idx], fileh.root.mouth[
                                                                             start_idx:end_idx], fileh.root.label[
                                                                                                 start_idx:end_idx]
                # mean subtraction
                speech = (speech - mean_speech) / 186.0
                mouth = (mouth - mean_mouth) / 255.0

                loss_value, score_dissimilarity, test_accuracy = sess.run([loss, distance_l2, accuracy],
                                              feed_dict={batch_speech: speech, batch_mouth: mouth,
                                                         batch_labels: label.reshape([FLAGS.batch_size, 1])})


                print("Epoch " + str(epoch + 1) + ", Minibatch " + str(i+1) + " of %d " % num_batches_per_epoch)
                score_dissimilarity_vector[start_idx:end_idx] = score_dissimilarity
                label_vector[start_idx:end_idx] = label

        # ROC

        ##############################
        ##### K-split validation #####
        ##############################
        K = 10
        EER = np.zeros((K, 1))
        AUC = np.zeros((K, 1))
        batch_k_validation = int(label_vector.shape[0] / float(K))

        for i in range(K):
            EER[i,:], AUC[i,:] = calculate_roc.calculate_eer_auc(label_vector[i * batch_k_validation:(i+1) * batch_k_validation], score_dissimilarity_vector[i * batch_k_validation:(i+1) * batch_k_validation])
        print('EER=',np.mean(EER,axis=0),np.std(EER,axis=0))
        print('STD=',np.mean(AUC,axis=0),np.std(AUC,axis=0))




if __name__ == '__main__':
    tf.app.run()
