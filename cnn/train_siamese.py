import tensorflow as tf
import numpy as np
from siamese import SiameseRNN



# Parameters
# ==================================================

tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_float("dropout", 0.5, "Dropout probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_string("training_files", "person_match.train2", "training file (default: None)")
tf.flags.DEFINE_integer("hidden_units", 50, "Number of hidden units in softmax regression layer (default:50)")
tf.flags.DEFINE_integer("max_sequence_length", 20, "Max sequence length")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_string("optimizer", "adam", "Training algorithm to use(sgd, adam, adadelta, ...")
tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate to use in the training algorithm")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.training_files==None:
    print "Input Files List is empty. use --training_files argument."
    exit()
    
#Model parameters
model_parameters = {}
model_parameters['rnn_type'] = FLAGS.rnn_type
model_parameters['dropout'] = FLAGS.dropout
model_parameters['padding'] = 'right'
model_parameters['sequence_length'] = FLAGS.max_sequence_length
model_parameters['opt'] = FLAGS.optimizer
model_parameters['learning_rate'] = FLAGS.learning_rate

    

        
        
        
        
    
    
    
    
    
    
    