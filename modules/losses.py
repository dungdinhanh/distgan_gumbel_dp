import numpy as np
import tensorflow as tf

def compute_nll_normal(pred, target, axis = None):

	c = -0.5 * tf.log(2 * np.pi)
	multiplier = 1.0 / (2.0 * 1)
        tmp = tf.square(pred - target)
	tmp *= -multiplier
	tmp += c

	return tmp
	
def compute_kl_loss(mu, log_var):
    return -0.5 * tf.reduce_sum(1 + log_var - tf.pow(mu, 2) - tf.exp(log_var))
