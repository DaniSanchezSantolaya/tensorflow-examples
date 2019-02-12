from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np
import cifar10_utils
import cifar10_siamese_utils
import convnet
import siamese

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = './checkpoints'



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
    Performs training and evaluation of ConvNet model.

    First define your graph using class ConvNet and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
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


        #Define graph using ConvNet class
        cnn = convnet.ConvNet(n_classes = num_classes)
        logits = cnn.inference(X)
        loss = cnn.loss(logits, y)
        acc = cnn.accuracy(logits, y)
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
            for step in range(FLAGS.max_steps):
                batch_x, batch_y = cifar10.train.next_batch(FLAGS.batch_size)
                # Run optimization 
                _, train_loss = sess.run([train_op, loss], feed_dict={X:batch_x, y:batch_y})



                if step % FLAGS.print_freq == 0 or step == FLAGS.max_steps - 1:
                    summary, train_acc = sess.run([merged,acc], feed_dict={X:batch_x, y:batch_y})
                    train_loss_history.append(train_loss)
                    train_acc_history.append(train_acc)
                    #print("Iteration {0:d}/{1:d}: Train Loss = {2:.3f}, Train Accuracy = {3:.3f}".format(
                        #step, FLAGS.max_steps, train_loss_history[-1], train_acc_history[-1]))
                    #summary = sess.run([merged])
                    train_writer.add_summary(summary, step)

                if step % FLAGS.eval_freq == 0 or step == FLAGS.max_steps - 1:
                    summary, test_loss, test_acc = sess.run([merged, loss, acc], feed_dict={X: x_test, y: y_test})
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


def train_siamese():
    """
    Performs training and evaluation of Siamese model.

    First define your graph using class Siamese and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    On train set, it is fine to monitor loss over minibatches. On the other
    hand, in order to evaluate on test set you will need to create a fixed
    validation set using the data sampling function you implement for siamese
    architecture. What you need to do is to iterate over all minibatches in
    the validation set and calculate the average loss over all minibatches.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    num_tuples = 1
    fraction_same = 0.2
    batch_size_test = 10000
    margin = 0.2
    cifar10 = cifar10_siamese_utils.get_cifar10(FLAGS.data_dir)
    num_classes = 10
    #Obtain validation set
    val_set = cifar10_siamese_utils.create_dataset(cifar10.test, num_tuples, batch_size_test, fraction_same)
    val_x1 = val_set[0][0]
    val_x2 = val_set[0][1]
    val_y = val_set[0][2]
    
        
    train_loss_history = []
    test_loss_history = []



    # Launch the graph
    with tf.Graph().as_default() as g:
        with tf.variable_scope('siamese') as scope:
            # Define placeholders for input
            X1 = tf.placeholder(tf.float32, shape=(None, cifar10.train.images.shape[1], cifar10.train.images.shape[2], cifar10.train.images.shape[3]))
            X2 = tf.placeholder(tf.float32, shape=(None, cifar10.train.images.shape[1], cifar10.train.images.shape[2], cifar10.train.images.shape[3]))
            y = tf.placeholder(tf.float32, shape=(None)) 


            #Define graph using ConvNet class
            cnn = siamese.Siamese()
            #Channel 1
            logits1 = cnn.inference(X1, reuse = False)
            #Channel 2
            logits2 = cnn.inference(X2, reuse = True)
            
            #Loss and training
            loss = cnn.loss(logits1, logits2, y, margin)
            train_op = train_step(loss)

            # Initializing the variables
            init = tf.initialize_all_variables()

            # Create a saver.
            saver = tf.train.Saver()
            
            with tf.Session() as sess:
                merged = tf.merge_all_summaries()
                train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train', sess.graph)
                test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test')
                sess.run(init)
                for step in range(FLAGS.max_steps):
                    batch_x1, batch_x2, batch_y = cifar10.train.next_batch(FLAGS.batch_size, fraction_same)
                    # Run optimization 
                    _ = sess.run(train_op, feed_dict={X1:batch_x1, X2:batch_x2, y:batch_y})


                    if step % FLAGS.print_freq == 0 or step == FLAGS.max_steps - 1:
                        summary, train_loss = sess.run([merged,loss], feed_dict={X1:batch_x1, X2:batch_x2, y:batch_y})
                        train_loss_history.append(train_loss)
                        #print("Iteration {0:d}/{1:d}: Train Loss = {2:.3f}, Train Accuracy = {3:.3f}".format(
                            #step, FLAGS.max_steps, train_loss_history[-1], train_acc_history[-1]))
                        #summary = sess.run([merged])
                        train_writer.add_summary(summary, step)

                    if step % FLAGS.eval_freq == 0 or step == FLAGS.max_steps - 1:
                        summary, test_loss = sess.run([merged, loss], feed_dict={X1: val_x1, X2: val_x2, y: val_y})
                        test_writer.add_summary(summary, step)
                        test_loss_history.append(test_loss)
                        print("Iteration {0:d}/{1:d}. Train Loss = {2:.3f}".format(
                                step, FLAGS.max_steps, train_loss_history[-1]))
                        print("Iteration {0:d}/{1:d}. Test Loss = {2:.3f}".format(
                                step, FLAGS.max_steps, test_loss_history[-1]))

          

                    if step % FLAGS.checkpoint_freq == 0 or step == FLAGS.max_steps - 1:
                        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)

        
    print("Best(min) test loss: " + str(min(test_loss_history)))
    ########################
    # END OF YOUR CODE    #
    ########################


def feature_extraction():
    """
    This method restores a TensorFlow checkpoint file (.ckpt) and rebuilds inference
    model with restored parameters. From then on you can basically use that model in
    any way you want, for instance, feature extraction, finetuning or as a submodule
    of a larger architecture. However, this method should extract features from a
    specified layer and store them in data files such as '.h5', '.npy'/'.npz'
    depending on your preference. You will use those files later in the assignment.

    Args:
        [optional]
    Returns:
        None
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################

    if FLAGS.train_model == 'linear':
        layers = ['fc1','fc2','flatten']
        features_path = './features/task1/noreg'    
    elif FLAGS.train_model == 'siamese':
        layers = ['L2-norm']
        features_path = './features/task2/'
    
    
    
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    num_classes = 10
    #Obtain test images
    x_test, y_test = cifar10.test.images, cifar10.test.labels
    x_train, y_train = cifar10.train.images, cifar10.train.labels
    labels_file_train = os.path.join('./features', 'labels_train.npy')
    labels_file_test = os.path.join('./features', 'labels_test.npy')
    np.save(labels_file_train, cifar10.train.labels)
    np.save(labels_file_test, cifar10.test.labels)

    
    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt-9999')
    checkpoint_meta_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt-9999.meta')


    # Launch the graph
    with tf.Graph().as_default() as g:
        # Define placeholders for input
        X = tf.placeholder(tf.float32, shape=(None, cifar10.train.images.shape[1], cifar10.train.images.shape[2], cifar10.train.images.shape[3]))
        y = tf.placeholder(tf.float32, shape=(None, num_classes)) 
        

        #Define graph
        if FLAGS.train_model == 'linear':
            cnn = convnet.ConvNet(n_classes = num_classes)
            logits = cnn.inference(X)
            loss = cnn.loss(logits, y)
            acc = cnn.accuracy(logits, y)
            train_op = train_step(loss)
        elif FLAGS.train_model == 'siamese':
            with tf.variable_scope('siamese') as scope:
                cnn = siamese.Siamese()
                logits1 = cnn.inference(X, reuse = False)
                #logits2 = cnn.inference(X, reuse = True)


        # Create a saver.
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)
   
            #Obtain features train images - divide in batches to fit in memory
            for i in range(5):
                batch_x, batch_y = cifar10.train.next_batch(10000)
                for layer in layers:
                    if layer == 'fc2':
                        name_layer = 'ConvNet/fc2/f2_output:0'
                    elif layer == 'flatten':
                        name_layer = 'ConvNet/flatten/Flatten/Reshape:0'
                    elif layer == 'fc1':
                        name_layer = 'ConvNet/fc1/f1_output:0'
                    if layer == 'L2-norm':
                        name_layer = 'siamese/ConvNet/L2-norm/L2-norm-output:0'
                    hidden_layer = sess.graph.get_tensor_by_name(name_layer)
                    hidden_layer_train = sess.run(hidden_layer, feed_dict={X: batch_x, y: batch_y})
                    features_file_train = os.path.join(features_path, 'features_' + str(layer) + '_train_' + str(i)+ '.npy')
                    np.save(features_file_train, hidden_layer_train)
            #Obtain features test images
            for layer in layers:
                if layer == 'fc2':
                    name_layer = 'ConvNet/fc2/f2_output:0'
                elif layer == 'flatten':
                    name_layer = 'ConvNet/flatten/Flatten/Reshape:0'
                elif layer == 'fc1':
                    name_layer = 'ConvNet/fc1/f1_output:0'
                if layer == 'L2-norm':
                    name_layer = 'siamese/ConvNet/L2-norm/L2-norm-output:0'
                hidden_layer = sess.graph.get_tensor_by_name(name_layer)
                hidden_layer_test = sess.run(hidden_layer, feed_dict={X: x_test, y: y_test})
                print(hidden_layer_test.shape)
                features_file_test = os.path.join(features_path, 'features_' + str(layer) + '_test.npy')                
                np.save(features_file_test, hidden_layer_test)


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

    if FLAGS.is_train:
        if FLAGS.train_model == 'linear':
            train()
        elif FLAGS.train_model == 'siamese':
            train_siamese()
        else:
            raise ValueError("--train_model argument can be linear or siamese")
    else:
        feature_extraction()

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
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                      help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                      help='Checkpoint directory')
    parser.add_argument('--is_train', type = bool, default = False,
                      help='Training or feature extraction')
    parser.add_argument('--train_model', type = str, default = 'linear',
                      help='Type of model. Possible options: linear and siamese')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
