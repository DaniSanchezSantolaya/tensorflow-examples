from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np
import vgg as v
import cifar10_utils

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'
REFINE_AFTER_K_STEPS_DEFAULT = 0

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = './checkpoints'


def create_var(name, shape, var_initializer, var_regularizer):
        var = tf.get_variable(name, shape, initializer=var_initializer, regularizer=var_regularizer)
        return var

def get_loss(logits, labels):
    cross_ent_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
    reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = tf.add(cross_ent_loss,reg_loss)
    tf.scalar_summary('Cross Entropy', cross_ent_loss)
    tf.scalar_summary('Regularization loss', reg_loss)
    tf.scalar_summary('Full loss', loss)
    return loss

def get_accuracy(logits, labels):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1), tf.argmax(labels,1)), tf.float32))
    tf.scalar_summary('accuracy', accuracy)
    return accuracy

def train_step(loss):
    """
    Defines the ops to conduct an optimization step. You can set a learning
    rate scheduler or pick your favorite optimizer here. This set of operations
    should be applicable to both ConvNet() and Siamese() objects.

    Args:
        loss: scalar float Tensor, full loss = cross_entropy + reg_loss

    Returns:
        train_op: Ops for optimization.
    """
    ########################
    # PUT YOUR CODE HERE  #
    ########################
    OPTIMIZER_DICT = {'sgd': tf.train.GradientDescentOptimizer, # Gradient Descent
                  'adadelta': tf.train.AdadeltaOptimizer, # Adadelta
                  'adagrad': tf.train.AdagradOptimizer, # Adagrad
                  'ADAM': tf.train.AdamOptimizer, # Adam
                  'rmsprop': tf.train.RMSPropOptimizer # RMSprop
                  } 
    train_op = OPTIMIZER_DICT[OPTIMIZER_DEFAULT](learning_rate=FLAGS.learning_rate).minimize(loss)
    ########################
    # END OF YOUR CODE    #
    ########################

    return train_op


def train():
    """
    Performs training and evaluation of your model.

    First define your graph using vgg.py with your fully connected layer.
    Then define necessary operations such as trainer (train_step in this case),
    savers and summarizers. Finally, initialize your model within a
    tf.Session and do the training.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every PRINT_FREQ iterations
    - on test set every EVAL_FREQ iterations

    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    num_classes = 10
    #Obtain test images
    x_test, y_test = cifar10.test.images, cifar10.test.labels
        
    train_acc_history = []
    train_loss_history = []
    test_loss_history = []
    test_acc_history = []


    # Launch the graph
    with tf.Graph().as_default() as g:
        # Define placeholders for input
        X = tf.placeholder(tf.float32, shape=(None, cifar10.train.images.shape[1], cifar10.train.images.shape[2], cifar10.train.images.shape[3]))
        y = tf.placeholder(tf.float32, shape=(None, num_classes)) 
        k = tf.placeholder(tf.int32)

        #Define graph 
        vgg_pool5, assign_ops = v.load_pretrained_VGG16_pool5(X, scope_name='vgg')
        #vgg_pool5 = tf.stop_gradients(vgg_pool5)
        vgg_pool5 = tf.cond(k >= FLAGS.refine_after_k, lambda: vgg_pool5, lambda: tf.stop_gradient(vgg_pool5))
        #flatten
        with tf.variable_scope('flatten') as scope:
            flatten = tf.contrib.layers.flatten(vgg_pool5)

        #Initializers
        weights_initializer = tf.random_normal_initializer(stddev=0.01)
        #weights_initializer = tf.contrib.layers.xavier_initializer()
        bias_initializer = tf.constant_initializer(0.0)
        #Regularizations
        weights_regularization = tf.contrib.layers.l2_regularizer(0.05)
        #fc1
        with tf.variable_scope('fc1') as scope:
            fc1_input_dim = flatten.get_shape()[1].value
            fc1_output_dim = 384
            weights = create_var('weights', [fc1_input_dim, fc1_output_dim], 
                                var_initializer=weights_initializer, var_regularizer=weights_regularization)         
            bias = create_var('bias', [fc1_output_dim], var_initializer=bias_initializer, var_regularizer=None)
            fc1_relu = tf.nn.relu(tf.matmul(flatten, weights) + bias, name='f1_output')
            
        #fc2
        with tf.variable_scope('fc2') as scope:
            fc2_input_dim = 384
            fc2_output_dim = 192
            weights = create_var('weights', [fc2_input_dim, fc2_output_dim], 
                                        var_initializer=weights_initializer, var_regularizer=weights_regularization)
            bias = create_var('bias', [fc2_output_dim], var_initializer=bias_initializer, var_regularizer=None)
            fc2_relu = tf.nn.relu(tf.matmul(fc1_relu, weights) + bias, name='f2_output')


        #fc3
        with tf.variable_scope('fc3') as scope:
            fc3_input_dim = 192
            fc3_output_dim = num_classes
            weights = create_var('weights', [fc3_input_dim, fc3_output_dim],
                                            var_initializer=weights_initializer, var_regularizer=None)
            bias = create_var('bias', [fc3_output_dim], var_initializer=bias_initializer, var_regularizer=None)
            logits = tf.add(tf.matmul(fc2_relu, weights), bias, name='logits')

        loss = get_loss(logits, y)
        acc = get_accuracy(logits, y)
        train_op = train_step(loss)


        # Initializing the variables
        init = tf.initialize_all_variables()

        # Create a saver.
        #saver = tf.train.Saver(tf.all_variables())
        saver = tf.train.Saver()
        with tf.Session() as sess:
            merged = tf.merge_all_summaries()
            train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train', sess.graph)
            test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test')
            sess.run(init)
            sess.run(assign_ops)
            for step in range(FLAGS.max_steps):
                batch_x, batch_y = cifar10.train.next_batch(FLAGS.batch_size)
                # Run optimization 
                _, train_loss = sess.run([train_op, loss], feed_dict={X:batch_x, y:batch_y, k:step})


                if step % FLAGS.print_freq == 0 or step == FLAGS.max_steps - 1:
                    summary, train_acc = sess.run([merged,acc], feed_dict={X:batch_x, y:batch_y, k:step})
                    train_loss_history.append(train_loss)
                    train_acc_history.append(train_acc)
                    #print("Iteration {0:d}/{1:d}: Train Loss = {2:.3f}, Train Accuracy = {3:.3f}".format(
                        #step, FLAGS.max_steps, train_loss_history[-1], train_acc_history[-1]))
                    #summary = sess.run([merged])
                    train_writer.add_summary(summary, step)

                if step % FLAGS.eval_freq == 0 or step == FLAGS.max_steps - 1:
                    summary, test_loss, test_acc = sess.run([merged, loss, acc], feed_dict={X: x_test, y: y_test, k:step})
                    test_writer.add_summary(summary, step)
                    test_loss_history.append(test_loss)
                    test_acc_history.append(test_acc)
                    print("Iteration {0:d}/{1:d}. Test Loss = {2:.3f}, Test Accuracy = {3:.3f}".format(
                            step, FLAGS.max_steps, test_loss_history[-1], test_acc_history[-1]))
      

                if step % FLAGS.checkpoint_freq == 0 or step == FLAGS.max_steps - 1:
                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

        
    print("Best test accuracy: " + str(max(test_acc_history)))
    ########################
    # END OF YOUR CODE    #
    ########################

def initialize_folders():
    """
    Initializes all folders in FLAGS variable.
    """

    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MakeDirs(FLAGS.data_dir)

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main(_):
    print_flags()

    initialize_folders()
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
    parser.add_argument('--print_freq', type = int, default = PRINT_FREQ_DEFAULT,
                      help='Frequency of evaluation on the train set')
    parser.add_argument('--eval_freq', type = int, default = EVAL_FREQ_DEFAULT,
                      help='Frequency of evaluation on the test set')
    parser.add_argument('--refine_after_k', type = int, default = REFINE_AFTER_K_STEPS_DEFAULT,
                      help='Number of steps after which to refine VGG model parameters (default 0).')
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                      help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                      help='Checkpoint directory')


    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
