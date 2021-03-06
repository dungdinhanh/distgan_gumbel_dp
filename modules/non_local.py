# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os, sys

sys.path.append('/home/mangroup/Documents/Code/Differential_Privacy/distgandp/modules/')

import tensorflow as tf
import numpy as np
from sagan_ops import ops



top_scope = tf.get_variable_scope()

def conv1x1(input_, output_dim,
            init=tf.contrib.layers.xavier_initializer(), name='conv1x1'):
  k_h = 1
  k_w = 1
  d_h = 1
  d_w = 1
  with tf.variable_scope(name):
    w = tf.get_variable(
        'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=init)
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
    return conv

def sn_conv1x1(input_, output_dim, update_collection,
              init=tf.contrib.layers.xavier_initializer(), name='sn_conv1x1'):
  with tf.variable_scope(name):
    k_h = 1
    k_w = 1
    d_h = 1
    d_w = 1
    w = tf.get_variable(
        'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=init)
    w_bar = ops.spectral_normed_weight(w, num_iters=1, update_collection=update_collection)

    conv = tf.nn.conv2d(input_, w_bar, strides=[1, d_h, d_w, 1], padding='SAME')
    return conv

def sn_non_local_block_sim(x, update_collection, name, init=tf.contrib.layers.xavier_initializer()):
  with tf.variable_scope(name):
    batch_size, h, w, num_channels = x.get_shape().as_list()
    location_num = h * w
    downsampled_num = location_num // 4

    # theta path
    theta = sn_conv1x1(x, num_channels // 8, update_collection, init, 'sn_conv_theta')
    theta = tf.reshape(
        theta, [batch_size, location_num, num_channels // 8])

    # phi path
    phi = sn_conv1x1(x, num_channels // 8, update_collection, init, 'sn_conv_phi')
    phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2)
    phi = tf.reshape(
        phi, [batch_size, downsampled_num, num_channels // 8])


    attn = tf.matmul(theta, phi, transpose_b=True)
    attn = tf.nn.softmax(attn)
    print(tf.reduce_sum(attn, axis=-1))

    # g path
    g = sn_conv1x1(x, num_channels // 2, update_collection, init, 'sn_conv_g')
    g = tf.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2)
    g = tf.reshape(
      g, [batch_size, downsampled_num, num_channels // 2])

    attn_g = tf.matmul(attn, g)
    attn_g = tf.reshape(attn_g, [batch_size, h, w, num_channels // 2])
    sigma = tf.get_variable(
        'sigma_ratio', [], initializer=tf.constant_initializer(0.0))
    sigma = tf.maximum(sigma, 0.0001)
    attn_g = sn_conv1x1(attn_g, num_channels, update_collection, init, 'sn_conv_attn')
    return x + sigma * attn_g


def non_local_block_sim(x, name, init=tf.contrib.layers.xavier_initializer()):
  #with tf.variable_scope(name):
    batch_size, h, w, num_channels = x.get_shape().as_list()
    location_num = h * w
    downsampled_num = location_num // 4

    # theta path
    theta = conv1x1(x, num_channels // 8, init, 'sn_conv_theta')
    theta = tf.reshape(
        theta, [batch_size, location_num, num_channels // 8])

    # phi path
    phi = conv1x1(x, num_channels // 8, init, 'sn_conv_phi')
    phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2)
    phi = tf.reshape(
        phi, [batch_size, downsampled_num, num_channels // 8])


    attn = tf.matmul(theta, phi, transpose_b=True)
    attn = tf.nn.softmax(attn, dim=-1) #ver 1.2, #newer: axis=1
    print(tf.reduce_sum(attn, axis=-1))

    # g path
    g = conv1x1(x, num_channels // 2, init, 'sn_conv_g')
    g = tf.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2)
    g = tf.reshape(
      g, [batch_size, downsampled_num, num_channels // 2])

    attn_g = tf.matmul(attn, g)
    attn_g = tf.reshape(attn_g, [batch_size, h, w, num_channels // 2])
    sigma = tf.get_variable(
        'sigma_ratio', [], initializer=tf.constant_initializer(0.0))
    sigma = tf.maximum(sigma, 0.0001)    
    attn_g = conv1x1(attn_g, num_channels, init, 'sn_conv_attn')
    return x + sigma * attn_g, attn

def non_local_block_sim_1(x, iteration, name, init=tf.contrib.layers.xavier_initializer()):
    
    batch_size, h, w, num_channels = x.get_shape().as_list()
    location_num = h * w
    downsampled_num = location_num//4

    # theta path
    theta = conv1x1(x, num_channels // 2, init, 'attetion_non_local_theta')
    theta = tf.reshape(
        theta, [batch_size, location_num, num_channels // 2])

    # phi path
    phi = conv1x1(x, num_channels // 2, init, 'attetion_non_local_phi')
    phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2)
    phi = tf.reshape(
        phi, [batch_size, downsampled_num, num_channels // 2])


    attn = tf.matmul(theta, phi, transpose_b=True)
    attn = tf.nn.softmax(attn, dim=-1) #ver 1.2, #newer: axis=1
    print(tf.reduce_sum(attn, axis=-1))

    # g path
    g = conv1x1(x, num_channels, init, 'attetion_non_local_g')
    g = tf.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2)
    g = tf.reshape(
      g, [batch_size, downsampled_num, num_channels])

    attn_g = tf.matmul(attn, g)
    attn_g = tf.reshape(attn_g, [batch_size, h, w, num_channels])
    #sigma = tf.get_variable(
    #    'attetion_non_local_sigma_ratio', [], initializer=tf.constant_initializer(0.00)) #0.05 seems ok
    #sigma = tf.maximum(sigma, 0.05)
    #sigma = 0.05
    #sigma = tf.maximum(0.0, 0.05 * (iteration - 0.5))
    sigma  = tf.cond(iteration <= 0.5, lambda: 0.0, lambda: 0.05 * (iteration - 0.5))
    #print('sigma = {}'.format(sigma))
    attn_g = conv1x1(attn_g, num_channels, init, 'attetion_non_local_attn')
    xshape = x.get_shape().as_list()
    x_out  = x + sigma * attn_g
    x_out  = tf.reshape(x_out, xshape)
    return x_out, attn, attn_g #converged but sigma is large: 1.30
    
def non_local_block_sim2(x, iteration, name, init=tf.contrib.layers.xavier_initializer()):

  # change this function, must change the other if using shared attention
  with tf.variable_scope(top_scope):
    return non_local_block_sim_1(x, iteration, name, init)
    
def non_local_block_sim3(x, iteration, name, init=tf.contrib.layers.xavier_initializer()):

  # change this function, must change the other if using shared attention
  with tf.variable_scope(top_scope, reuse=True):
    return non_local_block_sim_1(x, iteration, name, init)
