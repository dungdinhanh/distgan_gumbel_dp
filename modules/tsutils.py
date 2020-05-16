from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import tensorflow as tf

def softmax_to_onehot(x):
	x_max     = tf.nn.top_k(x).indices
	x_one_hot = tf.one_hot(x_max, tf.shape(x)[1])
	x_one_hot = tf.reshape(x_one_hot, tf.shape(x))
	return x_one_hot
	
