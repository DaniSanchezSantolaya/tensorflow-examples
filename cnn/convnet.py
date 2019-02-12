from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np



class ConvNet(object):
    """
   This class implements a convolutional neural network in TensorFlow.
   It incorporates a certain graph model to be trained and to be used
   in inference.
    """

    def __init__(self, n_classes = 10):
        """
        Constructor for an ConvNet object. Default values should be used as hints for
        the usage of each parameter.
        Args:
          n_classes: int, number of classes of the classification problem.
                          This number is required in order to specify the
                          output dimensions of the ConvNet.
        """
        self.n_classes = n_classes



    def _create_var(self, name, shape, var_initializer, var_regularizer):
        var = tf.get_variable(name, shape, initializer=var_initializer, regularizer=var_regularizer)
        return var

    def _add_summary(self, x):
      tf.histogram_summary(x.op.name + '/histogram', x)
      tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


    def inference(self, x):
        """
        Performs inference given an input tensor. This is the central portion
        of the network where we describe the computation graph. Here an input
        tensor undergoes a series of convolution, pooling and nonlinear operations
        as defined in this method. For the details of the model, please
        see assignment file.

        Here we recommend you to consider using variable and name scopes in order
        to make your graph more intelligible for later references in TensorBoard
        and so on. You can define a name scope for the whole model or for each
        operator group (e.g. conv+pool+relu) individually to group them by name.
        Variable scopes are essential components in TensorFlow for parameter sharing.
        Although the model(s) which are within the scope of this class do not require
        parameter sharing it is a good practice to use variable scope to encapsulate
        model.

        Args:
          x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]

        Returns:
          logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
                  the logits outputs (before softmax transformation) of the
                  network. These logits can then be used with loss and accuracy
                  to evaluate the model.
        """
        with tf.variable_scope('ConvNet'):
            ########################
            # PUT YOUR CODE HERE  #
            ########################
            #Initializers
            #filter_initializer = tf.contrib.layers.xavier_initializer()
            #weights_initializer = tf.contrib.layers.xavier_initializer()
            filter_initializer = tf.random_normal_initializer(stddev=0.01)
            weights_initializer = tf.random_normal_initializer(stddev=0.01)
            #filter_initializer = tf.random_uniform_initializer(minval =-0.01, maxval=0.01)
            #weights_initializer = tf.random_uniform_initializer(minval =-0.01, maxval=0.01)
            bias_initializer = tf.constant_initializer(0.0)
            #Regularizations
            weights_regularization = tf.contrib.layers.l2_regularizer(0.05)
            input_channels = x.get_shape()[3]
            #conv1
            with tf.variable_scope('conv1') as scope:
                filter1 = self._create_var('filter_weights', [5, 5, input_channels, 64], var_initializer=filter_initializer, var_regularizer=None) 
                conv1 = tf.nn.conv2d(x, filter1, [1, 1, 1, 1], padding='SAME', name=(scope.name + str('_conv')))
                biases1 = self._create_var('biases', [64], var_initializer=bias_initializer, var_regularizer=None)
                conv1 = tf.nn.bias_add(conv1, biases1)
                conv1_relu = tf.nn.relu(conv1, name=(scope.name + str('_relu')))

                conv1_maxpool = tf.nn.max_pool(conv1_relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=(scope.name + str('_maxpool')))
                #add sumaries
                #self._add_summary(filter1)
                #self._add_summary(conv1)
                #self._add_summary(conv1_relu)
                #self._add_summary(conv1_maxpool)
        

            
             #conv2
            with tf.variable_scope('conv2') as scope:
                filter2 = self._create_var('filter_weights', [5, 5, 64, 64], var_initializer=filter_initializer, var_regularizer=None)
                conv2 = tf.nn.conv2d(conv1_maxpool, filter2, [1, 1, 1, 1], padding='SAME', name=(scope.name + str('_conv')))
                biases2 = self._create_var('biases', [64], var_initializer=bias_initializer, var_regularizer=None)
                conv2 = tf.nn.bias_add(conv2, biases2)          
                conv2_relu = tf.nn.relu(conv2, name=(scope.name + str('_relu')))
                conv2_maxpool = tf.nn.max_pool(conv2_relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=(scope.name + str('_maxpool')))
                #add summaries
                #self._add_summary(filter2)
                #self._add_summary(conv2)
                #self._add_summary(conv2_relu)
                #self._add_summary(conv2_maxpool)



            #flatten
            with tf.variable_scope('flatten') as scope:
                flatten = tf.contrib.layers.flatten(conv2_maxpool)

            #fc1
            with tf.variable_scope('fc1') as scope:
                fc1_input_dim = flatten.get_shape()[1].value
                fc1_output_dim = 384
                weights = self._create_var('weights', [fc1_input_dim, fc1_output_dim], 
                                    var_initializer=weights_initializer, var_regularizer=weights_regularization)         
                bias = self._create_var('bias', [fc1_output_dim], var_initializer=bias_initializer, var_regularizer=None)
                fc1_relu = tf.nn.relu(tf.matmul(flatten, weights) + bias, name='f1_output')
                #Add summaries
                #self._add_summary(weights)
                #self._add_summary(bias)
                #self._add_summary(fc1_relu)

            #fc2
            with tf.variable_scope('fc2') as scope:
                fc2_input_dim = 384
                fc2_output_dim = 192
                weights = self._create_var('weights', [fc2_input_dim, fc2_output_dim], 
                                            var_initializer=weights_initializer, var_regularizer=weights_regularization)
                bias = self._create_var('bias', [fc2_output_dim], var_initializer=bias_initializer, var_regularizer=None)
                fc2_relu = tf.nn.relu(tf.matmul(fc1_relu, weights) + bias, name='f2_output')
                #Add summaries
                #self._add_summary(weights)
                #self._add_summary(bias)
                #self._add_summary(fc2_relu)

            #fc3
            with tf.variable_scope('fc3') as scope:
                fc3_input_dim = 192
                fc3_output_dim = self.n_classes
                weights = self._create_var('weights', [fc3_input_dim, fc3_output_dim],
                                                var_initializer=weights_initializer, var_regularizer=None)
                bias = self._create_var('bias', [fc3_output_dim], var_initializer=bias_initializer, var_regularizer=None)
                logits = tf.add(tf.matmul(fc2_relu, weights), bias, name='logits')
                #Add summaries
                #self._add_summary(weights)
                #self._add_summary(bias)
                #self._add_summary(logits)


            ########################
            # END OF YOUR CODE    #
            ########################
        return logits

    def accuracy(self, logits, labels):
        """
        Calculate the prediction accuracy, i.e. the average correct predictions
        of the network.
        As in self.loss above, you can use tf.scalar_summary to save
        scalar summaries of accuracy for later use with the TensorBoard.

        Args:
          logits: 2D float Tensor of size [batch_size, self.n_classes].
                       The predictions returned through self.inference.
          labels: 2D int Tensor of size [batch_size, self.n_classes]
                     with one-hot encoding. Ground truth labels for
                     each observation in batch.

        Returns:
          accuracy: scalar float Tensor, the accuracy of predictions,
                    i.e. the average correct predictions over the whole batch.
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1), tf.argmax(labels,1)), tf.float32))
        tf.scalar_summary('accuracy', accuracy)
        ########################
        # END OF YOUR CODE    #
        ########################

        return accuracy

    def loss(self, logits, labels):
        """
        Calculates the multiclass cross-entropy loss from the logits predictions and
        the ground truth labels. The function will also add the regularization
        loss from network weights to the total loss that is return.
        In order to implement this function you should have a look at
        tf.nn.softmax_cross_entropy_with_logits.
        You can use tf.scalar_summary to save scalar summaries of
        cross-entropy loss, regularization loss, and full loss (both summed)
        for use with TensorBoard. This will be useful for compiling your report.

        Args:
          logits: 2D float Tensor of size [batch_size, self.n_classes].
                       The predictions returned through self.inference.
          labels: 2D int Tensor of size [batch_size, self.n_classes]
                       with one-hot encoding. Ground truth labels for each
                       observation in batch.

        Returns:
          loss: scalar float Tensor, full loss = cross_entropy + reg_loss
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
        
        cross_ent_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = tf.add(cross_ent_loss,reg_loss)
        tf.scalar_summary('Cross Entropy', cross_ent_loss)
        tf.scalar_summary('Regularization loss', reg_loss)
        tf.scalar_summary('Full loss', loss)
       

        ########################
        # END OF YOUR CODE    #
        ########################

        return loss
