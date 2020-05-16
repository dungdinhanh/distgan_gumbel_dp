from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from modules.ops import *
from modules import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim
from functools import partial

from modules.categorical import *

fc = partial(ops.flatten_fully_connected, activation_fn=None, \
             weights_initializer=tf.contrib.layers.xavier_initializer())
relu  = tf.nn.relu
lrelu = partial(ops.leak_relu, leak=0.2)

'''
========================================================================
Translator to transform the prior distribution
========================================================================
'''
def translator(z, z_dim, dim, name = 'translator', reuse=True, training=True):
    y = z
    print("FCNET translator setup ---")
    with tf.variable_scope(name, reuse=reuse):
        y = lrelu(fc(y, dim * 2))
        y = lrelu(fc(y, dim * 4))
        y = lrelu(fc(y, dim * 2))
        logit = fc(y, z_dim)
        return logit
        
'''
========================================================================
FCNET FOR CARDFRAUD
========================================================================
'''        
    
def encoder_fcnet_cardfraud (img, x_shape, z_dim=30, dim=30, \
                             num_classes = None, labels = None, \
                             iteration = None,\
                             name = 'encoder', \
                             reuse=True, training=True):
                                     
    y = img
    print("FCNET encoder setup ---")
    with tf.variable_scope(name, reuse=reuse):
        y = relu(fc(y, dim))
        y = relu(fc(y, int(dim/2)))
        y = relu(fc(y, dim))
        logit = fc(y, z_dim)
        return logit    

'''
The generator
'''
def generator_fcnet_cardfraud(z, x_shape, dim=30, \
                       num_classes = None, labels = None,\
                       iteration = None,\
                       name = 'generator', \
                       reuse=True, training=True):
                           
    x_dim = x_shape[0] * x_shape[1]
    print("FCNET generator setup---")
    y = z
    with tf.variable_scope(name, reuse=reuse):
        y = relu(fc(y, dim))
        y = relu(fc(y, int(dim/2)))
        y = relu(fc(y, dim))
        y = fc(y, x_dim)
        y = tf.reshape(y, [-1, x_dim])
        return tf.nn.sigmoid(y)

def generator_fcnet_cardfraud_categorical(z, x_shape, dim=30, \
                       num_classes = None, labels = None,\
                       iteration = None,\
                       name = 'generator', \
                       categorical_softmax_use = 0, dtype = None, sdim = None,
                       reuse=True, training=True):
                           
    x_dim = x_shape[0] * x_shape[1]
    print("FCNET generator setup---")
    y = z
    y_concat = []
    with tf.variable_scope(name, reuse=reuse):
        y = relu(fc(y, dim))
        y = relu(fc(y, int(dim/2)))
        y = relu(fc(y, dim))
        '''
        # no softmax
        '''
        if categorical_softmax_use == 0 or categorical_softmax_use == 1:
           y = tf.sigmoid(fc(y, x_dim))
        else:
           # softmax
           # gumbel softmax parameters
           tau  = tf.Variable(1.0,name="temperature") # temperature
           hard = True
           for i in range(len(sdim)):
                if sdim[i] > 1:
                   # softmax
                   if categorical_softmax_use == 3: # gumbel-softmax
                      tmp = gumbel_softmax(fc(y, sdim[i]),tau, hard)
                   elif categorical_softmax_use == 2:
                      tmp = tf.nn.softmax(fc(y, sdim[i]),dim=-1)
                   else:
                       print('[models_fcnet.py -- generator_fcnet_cardfraud_categorical] categorical_softmax_use = %d is invalid.' % (categorical_softmax_use))
                       exit()
                   y_concat.append(tmp)
                else:
                   tmp = tf.nn.sigmoid(fc(y, sdim[i]))
                   y_concat.append(tmp)
           y = tf.concat(y_concat, axis=1)
           print('[models_fcnet.py -- generator_fcnet_cardfraud_categorical] categorical_softmax = %d used.' % (categorical_softmax_use))
           
        '''
        gumbel softmax
        '''
        
        y = tf.reshape(y, [-1, x_dim])
        return y
        
'''
The discriminator
'''               
def discriminator_fcnet_cardfraud(img, x_shape, dim=30, \
                             num_classes = None, labels = None,\
                             iteration = None,\
                             ssgan = 0, \
                             name='discriminator', \
                             reuse=True, training=True):
                                 
    print("FCNET discriminator setup---")
    y = img
   #  print(img)
   #  exit()
    with tf.variable_scope(name, reuse=reuse):
        y = relu(fc(y, dim))
        y = relu(fc(y, int(dim/2)))
        y = relu(fc(y, dim))
        feature = y
        logit = fc(y, 1)
        return tf.nn.sigmoid(logit),\
               logit,\
               tf.reshape(feature,[-1, dim])

'''
========================================================================
FCNET FOR CICIDS 2017
========================================================================
'''

'''
The encoder for CICIDS (vectors 1 x 190)
'''

def encoder_fcnet_cicids2017(img, x_shape, z_dim=128, dim=64, \
                             kernel_size=5, stride=2, \
                             num_classes = None, labels = None, \
                             attention = False,\
                             iteration = None,\
                             name = 'encoder', \
                             reuse=True, training=True):
                                     
    y = img
    dim = dim * 6
    print("FCNET encoder setup ---")
    with tf.variable_scope(name, reuse=reuse):
        y = relu(fc(y, dim * 4))
        y = relu(fc(y, dim * 4))
        y = relu(fc(y, dim * 4))
        y = relu(fc(y, dim * 8))
        y = relu(fc(y, dim * 8))
        y = relu(fc(y, dim * 8))
        y = relu(fc(y, dim * 16))
        y = relu(fc(y, dim * 8))
        y = relu(fc(y, dim * 8))
        y = relu(fc(y, dim * 8))
        y = relu(fc(y, dim * 4))
        y = relu(fc(y, dim * 4))
        y = relu(fc(y, dim * 4))      
        logit = fc(y, z_dim)
        return logit

'''
The generator for MNIST (28x28 images)
'''
def generator_fcnet_cicids2017(z, x_shape, dim=64, \
                       kernel_size=5, stride=2, \
                       num_classes = None, labels = None,\
                       attention = False,\
                       iteration = None,\
                       name = 'generator', \
                       reuse=True, training=True):
                           
    #bn = partial(batch_norm, is_training=training)
    #fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, \
    #                                            biases_initializer=None)
    x_dim = x_shape[0] * x_shape[1]
    print("FCNET generator setup---")
    y = z
    dim = dim * 6
    with tf.variable_scope(name, reuse=reuse):
        y = relu(fc(y, dim * 4))
        y = relu(fc(y, dim * 4))
        y = relu(fc(y, dim * 4))
        y = relu(fc(y, dim * 8))
        y = relu(fc(y, dim * 8))
        y = relu(fc(y, dim * 8))
        y = relu(fc(y, dim * 16))
        y = relu(fc(y, dim * 8))
        y = relu(fc(y, dim * 8))
        y = relu(fc(y, dim * 8))
        y = relu(fc(y, dim * 4))
        y = relu(fc(y, dim * 4))
        y = relu(fc(y, dim * 4))     
        y = fc(y, x_dim)
        y = tf.reshape(y, [-1, x_dim])
        return y

'''
The discriminator for MNIST (28x28 images)
'''               
def discriminator_fcnet_cicids2017(img, x_shape, dim=64, \
                             kernel_size=5, stride=2, \
                             num_classes = None, labels = None,\
                             attention = False, rotatelab = False,\
                             iteration = None,\
                             name='discriminator', \
                             reuse=True, training=True):
                                 
    print("FCNET discriminator setup---")
    y = img
    with tf.variable_scope(name, reuse=reuse):
        y = relu(fc(y, dim * 4))
        y = relu(fc(y, dim * 4))
        y = relu(fc(y, dim * 4))
        y = relu(fc(y, dim * 8))
        y = relu(fc(y, dim * 8))
        y = relu(fc(y, dim * 8))
        y = relu(fc(y, dim * 16))
        y = relu(fc(y, dim * 8))
        y = relu(fc(y, dim * 8))
        y = relu(fc(y, dim * 8))
        y = relu(fc(y, dim * 4))
        y = relu(fc(y, dim * 4))
        y = relu(fc(y, dim * 4))        
        feature = y
        logit = fc(y, 1)
        return tf.nn.sigmoid(logit),\
               logit,\
               tf.reshape(feature,[-1, dim * 4]), None
