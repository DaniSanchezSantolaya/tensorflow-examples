from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np



class Siamese(object):
    """
    This class implements a siamese convolutional neural network in
    TensorFlow. Term siamese is used to refer to architectures which
    incorporate two branches of convolutional networks parametrized
    identically (i.e. weights are shared). These graphs accept two
    input tensors and a label in general.
    """

    def _create_var(self, name, shape, var_initializer, var_regularizer, scope, reuse):
        if reuse:
            scope.reuse_variables()
        var = tf.get_variable(name, shape, initializer=var_initializer, regularizer=var_regularizer)
        return var

    def inference(self, x, reuse = False):
        """
        Defines the model used for inference. Output of this model is fed to the
        objective (or loss) function defined for the task.

        Here we recommend you to consider using variable and name scopes in order
        to make your graph more intelligible for later references in TensorBoard
        and so on. You can define a name scope for the whole model or for each
        operator group (e.g. conv+pool+relu) individually to group them by name.
        Variable scopes are essential components in TensorFlow for parameter sharing.
        You can use the variable scope to activate/deactivate 'variable reuse'.

        Args:
           x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]
           reuse: Python bool to switch reusing on/off.

        Returns:
           l2_out: L2-normalized output tensor of shape [batch_size, 192]

        Hint: Parameter reuse indicates whether the inference graph should use
        parameter sharing or not. You can study how to implement parameter sharing
        in TensorFlow from the following sources:

        https://www.tensorflow.org/versions/r0.11/how_tos/variable_scope/index.html
        """
        with tf.variable_scope('ConvNet') as conv_scope:
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
            weights_regularization = tf.contrib.layers.l2_regularizer(0.1)#None
            input_channels = x.get_shape()[3]
            #conv1
            with tf.variable_scope('conv1') as scope:
                filter1 = self._create_var('filter_weights', [5, 5, input_channels, 64], var_initializer=filter_initializer, var_regularizer=None, scope=scope, reuse=reuse) 
                conv1 = tf.nn.conv2d(x, filter1, [1, 1, 1, 1], padding='SAME', name=(scope.name + str('_conv')))
                biases1 = self._create_var('biases_test', [64], var_initializer=bias_initializer, var_regularizer=None, scope=scope, reuse=reuse)
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
                filter2 = self._create_var('filter_weights', [5, 5, 64, 64], var_initializer=filter_initializer, var_regularizer=None, scope=scope, reuse=reuse)
                conv2 = tf.nn.conv2d(conv1_maxpool, filter2, [1, 1, 1, 1], padding='SAME', name=(scope.name + str('_conv')))
                biases2 = self._create_var('biases', [64], var_initializer=bias_initializer, var_regularizer=None, scope=scope, reuse=reuse)
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
                                    var_initializer=weights_initializer, var_regularizer=weights_regularization, scope=scope, reuse=reuse)         
                bias = self._create_var('bias', [fc1_output_dim], var_initializer=bias_initializer, var_regularizer=None, scope=scope, reuse=reuse)
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
                                            var_initializer=weights_initializer, var_regularizer=weights_regularization, scope=scope, reuse=reuse)
                bias = self._create_var('bias', [fc2_output_dim], var_initializer=bias_initializer, var_regularizer=None, scope=scope, reuse=reuse)
                fc2_relu = tf.nn.relu(tf.matmul(fc1_relu, weights) + bias, name='f2_output')
                #Add summaries
                #self._add_summary(weights)
                #self._add_summary(bias)
                #self._add_summary(fc2_relu)

            #L2-norm
            with tf.variable_scope('L2-norm') as scope:
                l2_out = tf.nn.l2_normalize(fc2_relu, 1, name='L2-norm-output')

            
            ########################
            # END OF YOUR CODE    #
            ########################

        return l2_out

    def loss(self, channel_1, channel_2, label, margin):
        """
        Defines the contrastive loss. This loss ties the outputs of
        the branches to compute the following:

               L =  Y * d^2 + (1-Y) * max(margin - d^2, 0)

               where d is the L2 distance between the given
               input pair s.t. d = ||x_1 - x_2||_2 and Y is
               label associated with the pair of input tensors.
               Y is 1 if the inputs belong to the same class in
               CIFAR10 and is 0 otherwise.

               For more information please see:
               http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

        Args:
            channel_1: output of first channel (i.e. branch_1),
                              tensor of size [batch_size, 192]
            channel_2: output of second channel (i.e. branch_2),
                              tensor of size [batch_size, 192]
            label: Tensor of shape [batch_size]
            margin: Margin of the contrastive loss

        Returns:
            loss: scalar float Tensor
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################

        d_square = tf.reduce_sum(tf.square(channel_1 - channel_2), reduction_indices=1, keep_dims=True)
        contrastive_loss = label * d_square + (1 - label) * tf.maximum(margin - d_square, 0.)
        contrastive_loss = tf.reduce_mean(contrastive_loss)
        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = tf.add(contrastive_loss,reg_loss)
        tf.scalar_summary('Contrastive Loss', contrastive_loss)
        tf.scalar_summary('Regularization loss', reg_loss)
        tf.scalar_summary('Full loss', loss)

        ########################
        # END OF YOUR CODE    #
        ########################

        return loss
