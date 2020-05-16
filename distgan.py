# Ngoc-Trung Tran, 2018
# Tensorflow implementation of GAN models

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib as tc
import time

from modules.dataset import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
        
from modules.imutils import *
from modules.mdutils import *
from modules.vsutils import *
from modules.models_fcnet  import  *
from modules import ops
from modules.pie import *
from sys import exit



class DISTGAN(object):

    """
    Implementation of GAN methods.
    """

    def __init__(self, model='distgan', \
                 is_train = 0, \
                 lambda_p = 1.0, \
                 lambda_r = 1.0, \
                 lambda_w = 0.15625, \
                 lambda_d = 0.1,  \
                 lambda_g = 0.01, \
                 C     = 1., \
                 eps   = 1., \
                 delta = 1e-5, \
                 regc  = 2.5e-5, \
                 ncritics = 1, \
                 lr=2e-4, beta1 = 0.5, beta2 = 0.9, \
                 noise_dim = 100, \
                 prior_noise = 'uniform',\
                 nnet_type='dcgan', \
                 loss_type='log',\
                 df_dim = 64, gf_dim = 64, ef_dim = 64, \
                 gngan = 0, ssgan = 0, megan = 0, \
                 dataset = None, batch_size = 64, \
                 n_steps = 300000, \
                 decay_step = 10000, decay_rate = 1.0,\
                 log_interval=10, \
                 out_dir = './output/', \
                 verbose = True):
        """
        Initializing GAAN

        :param model: the model name
        :param lr:    learning rate 
        :param nnet_type: the network architecture type of generator, discrimintor, ...
        :dataset:     the dataset pointer
        :db_name:     database name obtained from dataset
        """
        
        tf.reset_default_graph()
        
        self.verbose      = verbose
        
        print('\n--- INFO ---')
        # dataset
        self.dataset   = dataset
        self.db_name   = self.dataset.db_name()
        print('* db_name = %s' % (self.db_name))
        self.categorical_softmax_use = self.dataset.get_categorical_softmax_use()
        if self.categorical_softmax_use > 0:
            self.dtype, self.sdim = self.dataset.get_categorical_info()

        # training parameters
        self.model      = model
        self.is_train   = is_train
        self.lr         = lr
        self.beta1      = beta1
        self.beta2      = beta2
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.n_steps    = n_steps
        self.batch_size = self.dataset.mb_size()
        
        if self.verbose == True:
            print('* model = %s, lr = %s, beta1 = %f, beta2 = %f, decay_step = %d, decay_rate = %f' % (self.model, self.lr, self.beta1, self.beta2, self.decay_step, self.decay_rate))
            print('* n_steps = %d, batch_size = %d' % (self.n_steps, self.batch_size))

        # architecture
        self.nnet_type = nnet_type
        self.loss_type = loss_type
        self.ef_dim    = ef_dim
        self.gf_dim    = gf_dim
        self.df_dim    = df_dim
        
        if self.verbose == True:
            print('* nnet_type = %s, loss_type = %s' % (self.nnet_type, self.loss_type))
            print('* ef_dim = %d, gf_dim = %d, df_dim = %d' % (self.ef_dim, self.gf_dim, self.df_dim))
        
        # new constraints
        self.gngan     = gngan
        self.ssgan     = ssgan
        self.megan     = megan
        
        if self.verbose == True:
            print('* gngan = %d, ssgan = %d, megan = %d' % (self.gngan, self.ssgan, self.megan))

        # dimensions
        self.data_dim   = dataset.data_dim()
        self.data_shape = dataset.data_shape()
        self.noise_dim  = noise_dim
        self.prior_noise = prior_noise
        
        if self.verbose == True:
            print('* data_dim = %d, noise_dim = %d, prior_noise: %s' % (self.data_dim, self.noise_dim, self.prior_noise))
            print('* data_shape = {}'.format(self.data_shape))

        # pamraeters
        self.lambda_p  = lambda_p
        self.lambda_r  = lambda_r
        self.lambda_w  = lambda_w
        self.lambda_d  = lambda_d
        self.lambda_g  = lambda_g
        
        if self.verbose == True:
            print('* lambda_p = %f, lambda_r = %f, lambda_w = %f, lambda_d = %f, lambda_g = %f' % (self.lambda_p, self.lambda_r, self.lambda_w, self.lambda_d, self.lambda_g))
        
        # differential privacy parameters
        self.C      = C
        self.eps    = eps
        self.delta  = delta
        self.regc   = regc
        # compute sigma
        self.q      = self.batch_size * 1.0 / self.dataset.data_size()
        self.sigma  = 2 * self.q * np.sqrt(np.log(1.0/self.delta)) / self.eps
        self.stddev = self.sigma * self.C
        
        if self.verbose == True:
            print('* C = %f, eps = %f, q = %f, delta = %f, sigma = %f, stddev = %f' % (self.C, self.eps, self.q, self.delta, self.sigma, self.stddev))
            
        self.nb_test_real = 10000
        self.nb_test_fake = 5000
        
        if self.verbose == True:
            print('* FID: nb_test_real = %d, nb_test_fake = %d' % ( self.nb_test_real, self.nb_test_fake ))

        # others
        self.out_dir      = out_dir
        self.ckpt_dir     = out_dir + '/model/'
        self.log_file     = out_dir + '.txt'
        self.log_interval = log_interval
                
        if self.verbose == True:
            print('* out_dir = {}'.format(self.out_dir))
            print('* ckpt_dir = {}'.format(self.ckpt_dir))
            print('* log_interval = {}'.format(self.log_interval))
            print('* verbose = {}'.format(self.verbose))
        
        print('--- END OF INFO ---')

        self.create_model()
        
        if self.db_name in ['mnist' , 'mnist_anomaly'] and self.noise_dim == 2:
            # Train classifier for MNIST to visualize latent space
            from modules.dataset import Dataset
            self.Classifier = classify()
            dataset_classifier = Dataset(name='mnist', source='./data/mnist/')
            self.Classifier.TrainwithoutSave(dataset_classifier.db_source())

    def sample_z(self, N):
        if self.prior_noise == 'uniform':
            return np.random.uniform(-1.0,1.0,size=[N, self.noise_dim])
        elif self.prior_noise == 'gaussian':
            return np.random.normal(0.0,5.0,size=[N, self.noise_dim])
        else:
            print('Prior noise: {} is not supported.' . format(prior_noise))
            exit()
            
    def dp_noise(self, tensor, batch_size):
        '''add noise to tensor'''
        s = tensor.get_shape().as_list()  # get shape of the tensor
        rt = tf.random_normal(s, mean=0.0, stddev= self.stddev)
        t = tf.add(tensor, tf.scalar_mul((1.0 / batch_size), rt))
        return t
        
    def create_discriminator(self):
        if self.nnet_type == 'fcnet' and (self.db_name == 'creditcardfraud' or \
             self.db_name == 'uci_epileptic_seizure' or \
             self.db_name == 'cervical_cancer' or self.db_name == 'fire_department' or \
             self.db_name == 'fire_department_integer' or self.db_name == 'fire_department_categorical'):
            return discriminator_fcnet_cardfraud               
        else:
            print('The dataset are not supported by the network');
            
    def create_generator(self): 
        if self.nnet_type == 'fcnet' and (self.db_name == 'creditcardfraud' or \
             self.db_name == 'uci_epileptic_seizure' or \
             self.db_name == 'cervical_cancer' or self.db_name == 'fire_department' or \
             self.db_name == 'fire_department_integer' or self.db_name == 'fire_department_categorical'):
            if self.categorical_softmax_use == 0:
                return generator_fcnet_cardfraud
            else:
                return generator_fcnet_cardfraud_categorical
        else:
            print('The dataset are not supported by the network');
            
    def create_encoder(self):
        if self.nnet_type == 'fcnet' and (self.db_name == 'creditcardfraud' or \
             self.db_name == 'uci_epileptic_seizure' or \
             self.db_name == 'cervical_cancer' or self.db_name == 'fire_department' or \
             self.db_name == 'fire_department_integer' or self.db_name == 'fire_department_categorical'):
            return encoder_fcnet_cardfraud                                
        else:
            print('The dataset are not supported by the network');            

    def create_optimizer(self, loss, var_list, learning_rate, beta1, beta2):
        """Create the optimizer operation.

        :param loss: The loss to minimize.
        :param var_list: The variables to update.
        :param learning_rate: The learning rate.
        :param beta1: First moment hyperparameter of ADAM.
        :param beta2: Second moment hyperparameter of ADAM.
        :return: Optimizer operation.
        """
        return tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2).minimize(loss, var_list=var_list)    
        
    def create_optimizer_dp(self, loss, var_list, learning_rate, beta1, beta2):
        optimizer      = tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2)
        gradients_vars = optimizer.compute_gradients(loss, var_list=var_list)
        _n_layers      = len(gradients_vars)
        gradients_vars_dp = [(tf.divide(gv[0], tf.maximum(tf.constant(1.), tf.divide(tf.norm(gv[0]), tf.constant(self.C, tf.float32)))) \
                              + self.dp_noise(gv[0], self.batch_size), gv[1]) for _ii, gv in enumerate(gradients_vars) \
                              if (_ii < _n_layers-1) and (gv[0] is not None)] # Clipping and noising
                              
        optimizer_new = optimizer.apply_gradients(gradients_vars_dp)
        return optimizer_new

    def create_model(self):

        self.X   = tf.placeholder(tf.float32, shape=[self.batch_size, self.data_dim])
        self.z   = tf.placeholder(tf.float32, shape=[self.batch_size, self.noise_dim])
        self.z1k = tf.placeholder(tf.float32, shape=[None, self.noise_dim]) # to generate 1k images
        
        self.iteration = tf.placeholder(tf.int32, shape=None)

        # argument real samples
        if self.ssgan == 1 or self.ssgan == 2:
           self.Xarg, self.larg, self.ridx = tf_argument_image_rotation(self.X, self.data_shape)
        elif self.ssgan == 3:
           self.Xarg, self.larg, self.ridx = tf_argument_image_rotation_plus_fake(self.X, self.data_shape)
           
        

        with tf.variable_scope('encoder'):
           self.E   = self.create_encoder()
           self.z_e = self.E(self.X, self.data_shape, self.noise_dim, dim = self.ef_dim, reuse=False)

        # create generator
        with tf.variable_scope('generator'):
            self.G   = self.create_generator()
            
            if self.categorical_softmax_use > 0:
                self.X_f = self.G(self.z,   self.data_shape, dim = self.gf_dim, categorical_softmax_use = self.categorical_softmax_use, dtype = self.dtype, sdim = self.sdim, reuse=False)   #generate fake samples
                
                self.X_r = self.G(self.z_e, self.data_shape, dim = self.gf_dim, categorical_softmax_use = self.categorical_softmax_use, dtype = self.dtype, sdim = self.sdim, reuse=True)    #generate reconstruction samples
                    
                self.X_f1k = self.G(self.z1k, self.data_shape, dim = self.gf_dim, categorical_softmax_use = self.categorical_softmax_use, dtype = self.dtype, sdim = self.sdim, reuse=True)  #generate 1k samples
                
                # to display
                self.X_f_img = self.G(self.z, self.data_shape, dim = self.gf_dim, categorical_softmax_use = self.categorical_softmax_use, dtype = self.dtype, sdim = self.sdim, reuse=True)  #generate reconstruction samples
            else:
                self.X_f = self.G(self.z,   self.data_shape, dim = self.gf_dim, reuse=False)   #generate fake samples
                
                self.X_r = self.G(self.z_e, self.data_shape, dim = self.gf_dim, reuse=True)    #generate reconstruction samples
                    
                self.X_f1k = self.G(self.z1k, self.data_shape, dim = self.gf_dim, reuse=True)  #generate 1k samples
                
                self.X_f_img = self.G(self.z, self.data_shape, dim = self.gf_dim, reuse=True)  #generate reconstruction samples             
                
            # argument fake samples
            if self.ssgan == 1 or self.ssgan == 2:
               self.Xarg_f, self.larg_f, _ = tf_argument_image_rotation(self.X_f, self.data_shape, self.ridx)
            elif self.ssgan == 3:
               self.Xarg_f, self.larg_f, _ = tf_argument_image_rotation_plus_fake(self.X_f,  self.data_shape, self.ridx)
        
        # argument real + fake samples
        if self.ssgan == 3:
            self.Xarg_mix, self.larg_mix, _ = tf_argument_image_rotation_and_fake_mix(self.X, self.X_f, self.data_shape)
        
        # create discriminator
        with tf.variable_scope('discriminator'):
            self.D   = self.create_discriminator()
            if self.ssgan == 1 or self.ssgan == 2 or self.ssgan == 3:
                self.d_real_sigmoid,  self.d_real_logit,  self.f_real,  _  = self.D(self.X,   self.data_shape, dim = self.df_dim, ssgan = self.ssgan, reuse=False)
                self.d_fake_sigmoid,  self.d_fake_logit,  self.f_fake,  _  = self.D(self.X_f, self.data_shape, dim = self.df_dim, ssgan = self.ssgan, reuse=True)
                self.d_recon_sigmoid, self.d_recon_logit, self.f_recon, _  = self.D(self.X_r, self.data_shape, dim = self.df_dim, ssgan = self.ssgan, reuse=True)
            
            else:
                self.d_real_sigmoid,  self.d_real_logit,  self.f_real  = self.D(self.X, self.data_shape, dim = self.df_dim, reuse=False)
                self.d_fake_sigmoid,  self.d_fake_logit,  self.f_fake  = self.D(self.X_f, self.data_shape, dim = self.df_dim, reuse=True)
                self.d_recon_sigmoid, self.d_recon_logit, self.f_recon = self.D(self.X_r, self.data_shape, dim = self.df_dim, reuse=True)
                
            # Compute gradient penalty
            epsilon = tf.random_uniform(shape=[tf.shape(self.X)[0],1], minval=0., maxval=1.)
            interpolation = epsilon * self.X + (1 - epsilon) * self.X_f
            if self.ssgan == 1 or self.ssgan == 2 or self.ssgan == 3:
                _,d_inter,_, _ = self.D(interpolation, self.data_shape, dim = self.df_dim, ssgan = self.ssgan, reuse=True)
            else:
                _,d_inter,_ = self.D(interpolation, self.data_shape, dim = self.df_dim, ssgan = self.ssgan, reuse=True)
            gradients = tf.gradients([d_inter], [interpolation])[0]
            slopes = tf.sqrt(tf.reduce_mean(tf.square(gradients), reduction_indices=[1]))
            self.penalty = tf.reduce_mean((slopes - 1) ** 2)

            # classifier with ssgan
            if self.ssgan == 1 or self.ssgan == 2:
                # predict real/fake classes
                _,  _,  _, self.real_cls = self.D(self.Xarg,   self.data_shape,   dim = self.df_dim, ssgan = self.ssgan, reuse=True)
                _,  _,  _, self.fake_cls = self.D(self.Xarg_f, self.data_shape,   dim = self.df_dim, ssgan = self.ssgan, reuse=True)

                # losses with SOFTMAX
                self.d_real_acc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.real_cls, labels=self.larg))
                  
                # losses with SOFTMAX
                self.d_acc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.real_cls, labels=self.larg))
                                
                # log loss for G
                self.g_real_acc = self.d_acc
                self.g_fake_acc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fake_cls, labels=self.larg_f))
                
                # hinge losses for G
                '''
                real_cls_softmax = tf.nn.softmax(self.real_cls, dim = -1)
                fake_cls_softmax = tf.nn.softmax(self.fake_cls, dim = -1)
                self.g_real_acc  = tf.reduce_mean(tf.multiply(real_cls_softmax, self.larg))
                self.g_fake_acc  = tf.reduce_mean(tf.multiply(fake_cls_softmax, self.larg_f))
                '''
                
                self.g_acc  = tf.abs(self.g_fake_acc - self.g_real_acc, name = 'abs') # our proposed ss loss
                #self.g_acc  = self.g_fake_acc # original ssloss
                
            elif self.ssgan == 3:
                # predict real/fake classes
                _,  _,  _, self.real_cls = self.D(self.Xarg,    self.data_shape, dim = self.df_dim, ssgan = self.ssgan, reuse=True)
                _,  _,  _, self.fake_cls = self.D(self.Xarg_f,  self.data_shape, dim = self.df_dim, ssgan = self.ssgan, reuse=True)
                _,  _,  _, self.mixe_cls = self.D(self.Xarg_mix,self.data_shape, dim = self.df_dim, ssgan = self.ssgan, reuse=True)
                
                # loss for D
                self.d_acc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.mixe_cls, labels=self.larg_mix))
                
                # losses for G
                self.g_real_acc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.real_cls, labels=self.larg))
                self.g_fake_acc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fake_cls, labels=self.larg_f))
                
                # hinge losses for G
                '''
                real_cls_softmax = tf.nn.softmax(self.real_cls, dim = -1)
                fake_cls_softmax = tf.nn.softmax(self.fake_cls, dim = -1)
                self.g_real_acc  = tf.reduce_mean(tf.multiply(real_cls_softmax, self.larg))
                self.g_fake_acc  = tf.reduce_mean(tf.multiply(fake_cls_softmax, self.larg_f))
                '''
                
                self.g_acc  = tf.abs(self.g_fake_acc - self.g_real_acc, name = 'abs')

        
        # reconstruction with regularization
        self.ae_loss = tf.reduce_mean(tf.square(self.f_real - self.f_recon))
        if self.gngan == 0 or self.gngan == 2: #eccv 2018
            #data-latent constraint
            self.md_x       = tf.reduce_mean(self.f_recon - self.f_fake)
            self.md_z       = tf.reduce_mean(self.z_e - self.z) * self.lambda_w
            self.ae_reg     = tf.square(self.md_x - self.md_z)
        elif self.gngan == 1 or self.gngan == 3: #aaai 2019
            #neighbor-embedding constraint
            z_mix = tf.concat([self.z_e, self.z], axis=0)
            f_mix = tf.concat([self.f_real, self.f_fake], axis=0)
            p = get_dist_table(f_mix, dist='tdis', \
                                             symmetric=False, alpha=1.)
            q = get_dist_table(z_mix, dist='tdis', \
                                             symmetric=False, alpha=1.)
            self.ae_reg     = get_KL_loss(p, q)
        else:
            print('Invalid gngan code. Exit!')
            exit()

        # Gradient matching (aaai 2019)
        if self.gngan == 2 or self.gngan == 3:
            grad_real = tf.gradients([self.d_real_logit], [self.X])[0]
            grad_real_norm = tf.sqrt(tf.reduce_mean(tf.square(grad_real), reduction_indices=[1]))
            grad_fake = tf.gradients([self.d_fake_logit], [self.X_f])[0]
            grad_fake_norm = tf.sqrt(tf.reduce_mean(tf.square(grad_fake), reduction_indices=[1]))
            self.grad_reg  = tf.square(tf.reduce_mean(grad_real_norm - grad_fake_norm))
                
            grad_real_dx = tf.multiply(grad_real, self.X)
            grad_real_dx_norm = tf.sqrt(tf.reduce_mean(tf.square(grad_real_dx), reduction_indices=[1]))
            grad_fake_dg = tf.multiply(grad_fake, self.X_f)
            grad_fake_dg_norm = tf.sqrt(tf.reduce_mean(tf.square(grad_fake_dg), reduction_indices=[1]))
            self.grad_reg_xdx = tf.square(tf.reduce_mean(grad_real_dx_norm - grad_fake_dg_norm))                
            
            
        # Decay the weight of reconstruction
        t = tf.cast(self.iteration, tf.float32)/self.n_steps
        # mu = 0 if t <= N/2, mu in [0,0.05] 
        # if N/2 < t and t < 3N/2 and mu = 0.05 if t > 3N/2
        self.mu = tf.maximum(tf.minimum((t*0.1-0.05)*2, 0.05),0.0)
        w_real  = 0.95 + self.mu
        w_recon = 0.05 - self.mu
        w_fake  = 1.0

        if self.loss_type == 'log':
            # Loss
            self.d_real  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_logit, labels=tf.ones_like(self.d_real_sigmoid)))
            self.d_fake  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_logit, labels=tf.zeros_like(self.d_fake_sigmoid)))
            self.d_recon = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_recon_logit,labels=tf.ones_like(self.d_recon_sigmoid)))

            # lower weights for d_recon to achieve sharper generated images, slightly improved from the original paper
            # self.d_cost_gan  = 0.95 * self.d_real + 0.05 * self.d_recon + self.d_fake + self.lambda_p * self.penalty #old (iccv 2019)
            self.d_cost_gan  = 0.95 * self.d_real + 0.05 * self.d_recon + self.d_fake + self.lambda_p * self.penalty #old (iccv 2019)            
            # self.d_cost = w_real * self.d_real + w_recon * self.d_recon + w_fake * self.d_fake + self.lambda_p * self.penalty #new
        elif self.loss_type == 'hinge':
            if self.nnet_type == 'dcgan':
                self.d_cost_gan = -(w_real * tf.reduce_mean(tf.minimum(0.,-1 + self.d_real_logit))  + \
                            w_recon * tf.reduce_mean(tf.minimum(0.,-1 + self.d_recon_logit)) + \
                            tf.reduce_mean(tf.minimum(0.,-1 - self.d_fake_logit)) + self.lambda_p * self.penalty)            
            else:
                self.d_cost_gan = -(w_real * tf.reduce_mean(tf.minimum(0.,-1 + self.d_real_sigmoid))  + \
                            w_recon * tf.reduce_mean(tf.minimum(0.,-1 + self.d_recon_sigmoid)) + \
                            tf.reduce_mean(tf.minimum(0.,-1 - self.d_fake_sigmoid)) + self.lambda_p * self.penalty)

                                  
        self.r_cost  = self.ae_loss + self.lambda_r * self.ae_reg
        self.g_cost_gan  = tf.abs(tf.reduce_mean(self.d_real_sigmoid - self.d_fake_sigmoid))
        
        # gradient matching of gngan
        if self.gngan == 2 or self.gngan == 3:
            self.g_cost = self.g_cost_gan + self.grad_reg + self.grad_reg_xdx
            
        # self-supervised learning of ssgan
        if self.ssgan == 1:
            self.d_cost = self.d_cost_gan + self.lambda_d * self.d_acc
            self.g_cost = self.g_cost_gan
        elif self.ssgan == 2:
            self.d_cost = self.d_cost_gan + self.lambda_d * self.d_acc
            self.g_cost = self.g_cost_gan + self.lambda_g * self.g_acc
        elif self.ssgan == 3:
            self.d_cost = self.d_cost_gan + self.lambda_d * self.d_acc
            self.g_cost = self.g_cost_gan + self.lambda_g * self.g_acc
        else:
            self.d_cost = self.d_cost_gan
            self.g_cost = self.g_cost_gan
        
        '''    
        # add regularization    
        self.reg = tc.layers.apply_regularization(
            tc.layers.l1_regularizer(self.regc),
            weights_list=[var for var in tf.all_variables() if 'weights' in var.name]
        )    
        
        self.d_cost = self.d_cost + self.reg
        self.g_cost = self.g_cost + self.reg
        self.r_cost = self.r_cost + self.reg
        '''
        

        # Create optimizers        
        if self.nnet_type == 'resnet':

            self.vars_g = [var for var in tf.trainable_variables() if 'generator' in var.name]
            print('[generator parameters]')
            print(self.vars_g) 
            self.vars_d = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
            print('[discriminator parameters]')
            print(self.vars_d)
            self.vars_e = [var for var in tf.trainable_variables() if 'encoder' in var.name]
            print('[encoder parameters]')
            print(self.vars_e)
            
            self.vars_g_save = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            self.vars_d_save = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            self.vars_e_save = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
            
            if self.is_train == 0:
                
                print('[decay the learning rate and network type: resnet]')
                self.decay_rate = tf.maximum(0., tf.minimum(1.-(tf.cast(self.iteration, tf.float32)/self.n_steps),0.5))
                
                self.opt_rec = tf.train.AdamOptimizer(learning_rate=self.lr * self.decay_rate, beta1=self.beta1, beta2=self.beta2)
                self.opt_gen = tf.train.AdamOptimizer(learning_rate=self.lr * self.decay_rate, beta1=self.beta1, beta2=self.beta2)
                self.opt_dis = tf.train.AdamOptimizer(learning_rate=self.lr * self.decay_rate, beta1=self.beta1, beta2=self.beta2)
                
                self.gen_gv  = self.opt_gen.compute_gradients(self.g_cost, var_list=self.vars_g)
                self.dis_gv  = self.opt_dis.compute_gradients(self.d_cost, var_list=self.vars_d)
                self.rec_gv  = self.opt_rec.compute_gradients(self.r_cost, var_list=self.vars_e) #relaxed reconstruction
                #self.rec_gv  = self.opt_rec.compute_gradients(self.r_cost, var_list=self.vars_e + self.vars_g) #strict reconstruction
                
                self.opt_r  = self.opt_rec.apply_gradients(self.rec_gv)    
                self.opt_g  = self.opt_gen.apply_gradients(self.gen_gv)
                self.opt_d  = self.opt_dis.apply_gradients(self.dis_gv)
            
        else:
            
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                # Create optimizers
                self.vars_e = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
                self.vars_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                self.vars_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                
                print('***[encoder parameters]***')
                print(self.vars_e)
                print('***[generator parameters]***')
                print(self.vars_g)
                print('***[discriminator parameters]***')
                print(self.vars_d)
                
                self.vars_e_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
                self.vars_g_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
                self.vars_d_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
                
                if self.is_train == 0:
                                                   
                    # Setup for weight decay
                    self.global_step = tf.Variable(0, trainable=False)
                    self.learning_rate = tf.train.exponential_decay(self.lr, self.global_step, self.decay_step, self.decay_rate, staircase=True)

                    if self.db_name in ['mnist', 'creditcardfraud']:
                        self.opt_r = self.create_optimizer(self.r_cost, self.vars_e + self.vars_g, self.learning_rate, self.beta1, self.beta2)
                    else:
                        self.opt_r = self.create_optimizer(self.r_cost, self.vars_e, self.learning_rate, self.beta1, self.beta2)
                    self.opt_g = self.create_optimizer(self.g_cost, self.vars_g, self.learning_rate, self.beta1, self.beta2)
                    
                    if self.C == 0:
                        # original optimizer
                        print('optimizing original D')
                        self.opt_d = self.create_optimizer(self.d_cost, self.vars_d, self.learning_rate, self.beta1, self.beta2)
                    else:
                        print('optimizing D with DP')
                        # optimizer for differential privacy
                        self.opt_d = self.create_optimizer_dp(self.d_cost, self.vars_d, self.learning_rate, self.beta1, self.beta2)
        
        self.init = tf.global_variables_initializer()

    def train(self):
        """
        Training the model
        """
        # d_steps = 5
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        
        fid = open(self.log_file,"w")
        
        saver = tf.train.Saver(var_list = self.vars_e_save + self.vars_g_save + self.vars_d_save, max_to_keep=300)
        
        #tf.set_random_seed(4)
        
        with tf.Session(config=run_config) as sess:
            
            start = time.time()
            sess.run(self.init)
                       
            for step in range(self.n_steps + 1):

                # train auto-encoder
                mb_X = self.dataset.next_batch()
                # print(mb_X.shape)
                # exit()
                mb_z = self.sample_z(np.shape(mb_X)[0])
                
                '''
                ano_idx = np.squeeze(np.where(mb_X[:,-1] == 1))
                if ano_idx.any():
                    print('Having training anomaly: {}'.format(ano_idx))
                '''
                
                if step == 0:
                    # check f_feature size of discriminator
                    # print(mb_X.shape)
                    # exit()
                    f_real = sess.run(self.f_real,feed_dict={self.X: mb_X, self.z: mb_z})
                    print('=== IMPORTANT !!!: SET CORRECT FEATURE SIZE: {} TO feature_dim OF MAIN FUNCTION ==='.format(np.shape(f_real)))
                
                                
                sess.run([self.opt_r],feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})

                # pre-train the network with diviersity
                # train discriminator
                mb_X = self.dataset.next_batch()
                mb_z = self.sample_z(np.shape(mb_X)[0])
                
                '''
                ano_idx = np.squeeze(np.where(mb_X[:,-1] == 1))
                if ano_idx.any():
                    print('Having training anomaly: {}'.format(ano_idx))
                '''
                sess.run([self.opt_d],feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})
                
                # train generator
                mb_X = self.dataset.next_batch()
                mb_z = self.sample_z(np.shape(mb_X)[0])
                sess.run([self.opt_g],feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})
                
                '''
                ano_idx = np.squeeze(np.where(mb_X[:,-1] == 1))
                if ano_idx.any():
                    print('Having training anomaly: {}'.format(ano_idx))
                '''

                # compute losses to print
                if self.ssgan > 0:
                    loss_d, loss_d_gan, loss_d_acc, loss_g, loss_g_gan, loss_g_acc, loss_r = sess.run([self.d_cost, self.d_cost_gan, self.d_acc, self.g_cost, self.g_cost_gan, self.g_acc, self.r_cost],feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})
                else:
                    loss_d, loss_g, loss_r = sess.run([self.d_cost, self.g_cost, self.r_cost],feed_dict={self.X: mb_X, self.z: mb_z, self.iteration: step})

                if step % self.log_interval == 0:
                    if self.verbose:
                            elapsed = int(time.time() - start)
                            if self.ssgan > 0:
                                output_str = 'step: {:4d}, D loss: {:8.4f}, D loss (gan): {:8.4f}, D loss (acc): {:8.4f} G loss: {:8.4f}, G loss (gan): {:8.4f}, G loss (acc): {:8.4f}, R loss: {:8.4f}, time: {:3d} s'.format(step, loss_d, loss_d_gan, loss_d_acc, loss_g, loss_g_gan, loss_g_acc, loss_r, elapsed)
                            else:
                                output_str = 'step: {:4d}, D loss: {:8.4f}, G loss: {:8.4f}, R loss: {:8.4f}, time: {:3d} s'.format(step, loss_d, loss_g, loss_r, elapsed)
                            print(output_str) 
                            #print('X_reg: %f' % (X_reg))
                            fid.write(str(output_str)+'\n')
                            fid.flush()

                if step % (self.log_interval*1000) == 0:
                    
                    if self.db_name in ['mnist', 'cifar10', 'stl10']:                 
                        if step == 0:
                            real_dir = self.out_dir + '/real/'
                            if not os.path.exists(real_dir):
                                os.makedirs(real_dir)
                                
                        fake_dir = self.out_dir + '/fake_%d/'%(step)
                        if not os.path.exists(fake_dir):
                            os.makedirs(fake_dir)
                            
                        #generate reals
                        if step == 0:
                            for v in range(self.nb_test_real // self.batch_size + 1):
                                #print(v, self.nb_test_real)
                                # train auto-encoder
                                mb_X = self.dataset.next_batch()
                                im_real_save = np.reshape(mb_X,(-1, self.data_shape[0], self.data_shape[1],self.data_shape[2]))
                                
                                for ii in range(np.shape(mb_X)[0]):
                                    real_path = real_dir + '/image_%05d.jpg' % (np.min([v*self.batch_size + ii, self.nb_test_real]))
                                    imwrite(im_real_save[ii,:,:,:], real_path)
                        elif step > 0:
                            #generate fake
                            for v in range(self.nb_test_fake // self.batch_size + 1):
                                #print(v, self.nb_test_fake)
                                mb_z = self.sample_z(np.shape(mb_X)[0])
                                im_fake_save = sess.run(self.X_f,feed_dict={self.z: mb_z})
                                im_fake_save = np.reshape(im_fake_save,(-1, self.data_shape[0], self.data_shape[1], self.data_shape[2]))

                                for ii in range(np.shape(mb_z)[0]):
                                    fake_path = fake_dir + '/image_%05d.jpg' % (np.min([v*self.batch_size + ii, self.nb_test_fake]))
                                    imwrite(im_fake_save[ii,:,:,:], fake_path)

                if step %10000==0:
                    if not os.path.exists(self.ckpt_dir +'%d/'%(step)):
                        os.makedirs(self.ckpt_dir +'%d/'%(step))
                    save_path = saver.save(sess, '%s%d/batch_%d.ckpt' % (self.ckpt_dir, step,step))
                    print('Model saved in file: % s' % save_path)

    def generate(self):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        with tf.Session(config=run_config) as sess:
            flag = load_checkpoint(self.ckpt_dir + '100000/', sess)
            if flag == True:
                print('Generating data ... ')
                noise = self.sample_z(1000)
                generated_data = sess.run(self.X_f1k, feed_dict = {self.z1k : noise})
                if(self.dataset.name in ['creditcardfraud', 'cervical_cancer']):    
                    generated_data[:,-1] = np.around(generated_data[:,-1])
                np.savetxt(self.ckpt_dir + self.db_name + "distgan_syn_data_%d.csv" % (self.n_steps), generated_data, delimiter=",")
            print(generated_data.shape)
        
        if(self.db_name == 'fire_department_integer'):
            df = pd.read_csv(self.dataset.source)
            with open('data/fire_department/2016-specs.json', 'r') as infile:
                specs = json.load(infile)
            columns = df.columns
            new_df = pd.DataFrame(data=generated_data, columns=columns)
            for column in columns:
                spec = specs[column]
                min_v = spec['min']
                max_v = spec['max']
                range_v = max_v - min_v
                real_data = new_df[column].values.astype('float')
                if(range_v!=0):
                    real_data = real_data*(1.0*range_v) + np.ones_like(real_data)*min_v
                else:
                    real_data = range_v
                new_df[column] = real_data
            new_df.astype(int).to_csv(self.ckpt_dir + self.db_name + "distgan_syn_data_%d.csv" % (self.n_steps), index=False)

    def post_process_categorical(self):
        sdim = self.dataset.sdim
        offset = []
        position = 0
        for dim in sdim:
            offset.append(position)
            position+=dim
        offset.append(sum(sdim))
        print(sdim)
        print(offset)
        print(self.dataset.source+'risk_factors_cervical_cancer.csv')
        source_df = pd.read_csv(self.dataset.source+'risk_factors_cervical_cancer.csv')
        columns = source_df.columns
        df = pd.read_csv(self.ckpt_dir + self.db_name + "distgan_syn_data_%d.csv" % (self.n_steps), index_col=False, header=None)
        # df = df.astype(int)
        data = df.values
        new_data = np.array([])
        for i in range(len(offset)-1):
            data_1 = data[:,range(offset[i],offset[i+1])]
            if(offset[i+1]-offset[i]>=2):
                data_1 = np.argmax(data_1, axis=1).reshape((-1,1))
            print(data_1.shape)
            if(new_data.size==0):
                new_data = data_1
            else:
                new_data = np.concatenate((new_data, data_1), axis=1)
        # new_data = new_data.astype(int)
        new_df = pd.DataFrame(data=new_data, columns=columns)
        print(new_df)
        # np.savetxt(self.ckpt_dir + self.db_name + "distgan_syn_data_%d.csv" % (self.n_steps), new_data, delimiter=",")
        new_df.astype(int).to_csv(self.ckpt_dir + self.db_name + "distgan_syn_data_%d.csv" % (self.n_steps), index=False)


        


    def eval_acc(self, classifier='LogisticRegression'):
        
        print('Evaluating the generated samples via accuracy ... ')
        
        real_data_train = self.dataset.load_train_data()
        real_data_test  = self.dataset.load_test_data()
        
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
                
        with tf.Session(config=run_config) as sess:
            flag = load_checkpoint(self.ckpt_dir + '/100000/', sess)
            if flag == True:
                gen_zero = []
                gen_one  = []
                
                print('Generating data ... ')
                
                nb_samples_factor = 1.
                positive_number = sum(real_data_test[:,-1]==1.0)
                negative_number = sum(real_data_test[:,-1]==0.0)
                print("positive_number", positive_number)
                print("negative_number", negative_number)

                while (len(gen_one) < positive_number * nb_samples_factor) or (len(gen_zero) < negative_number * nb_samples_factor):
                    noise = self.sample_z(len(real_data_test))
                    generated_data = sess.run(self.X_f1k, feed_dict = {self.z1k : noise})
                    generated_data = generated_data.reshape((-1, self.data_shape[0] * self.data_shape[1]))
                    generated_data = shuffle(generated_data.reshape(len(real_data_test), self.data_shape[0] * self.data_shape[1]))
                    generated_data[:,-1] = np.around(generated_data[:,-1])
                    for data in generated_data:
                        if data[-1] == 0.0:
                            if len(gen_zero) < negative_number * nb_samples_factor:
                                gen_zero.append(data)
                                print('# gen_zero = %d' % (len(gen_zero)))
                            else:
                                pass
                        elif data[-1] == 1.0:
                            if len(gen_one) < positive_number * nb_samples_factor:
                                gen_one.append(data)
                                print('# gen_one = %d' % (len(gen_one)))
                            else:
                                pass

                gen_zero = np.array(gen_zero)
                gen_one  = np.array(gen_one)
                print(np.shape(gen_zero))
                print(np.shape(gen_one))
                gen_data = np.concatenate((gen_zero,gen_one))
                gen_data = shuffle(gen_data)
                np.savetxt(self.ckpt_dir + str(self.dataset.name)+'_epsilon_%.5f_dpdistgan_gendata.csv' % (self.eps), gen_data, delimiter=",")
                
                print('generating feature histograms of original and gen data')
                draw_feat_hist(real_data_train, gen_data, self.out_dir + '/feat_hist/')
                
                print('n neg = %d, n pos = %d' % (negative_number, positive_number))
                
                print('---')
                                
                # Train the original data
                print('Training classifier on original data.')
                if(classifier=='LogisticRegression'):
                    lg_real = LogisticRegression(random_state=0, solver='saga').fit(real_data_train[:,:-1], real_data_train[:,-1]) #logistic regression
                elif(classifier=='MLPClassifier'):
                    lg_real = MLPClassifier(activation = 'tanh',hidden_layer_sizes = (18,18,18)).fit(real_data_train[:,:-1], real_data_train[:,-1]) #decision tree
                elif(classifier=='AdaBoostClassifier'):
                    lg_real = AdaBoostClassifier(base_estimator = LogisticRegression(),n_estimators = 200).fit(real_data_train[:,:-1], real_data_train[:,-1]) #adaboost
                elif(classifier=='BaggingClassifier'):
                    lg_real = BaggingClassifier(base_estimator = LogisticRegression(),n_estimators = 100).fit(real_data_train[:,:-1], real_data_train[:,-1]) #random forest
                else:
                    print("No Classifier")
                    exit()
                probs_real = lg_real.predict_proba(real_data_test[:,:-1])
                probs_real = probs_real[:, 1]
                auc_real = roc_auc_score(real_data_test[:,-1], probs_real)
                label_real = lg_real.predict(real_data_test[:,:-1])
                confusion_matrix_real = confusion_matrix(real_data_test[:,-1], label_real)
                tn, fp, fn, tp = confusion_matrix_real.ravel()
                print('tn = %f, fp = %f, fn = %f, tp = %f' % (tn, fp, fn, tp))
                print("auc_real", auc_real)
                print("true positive ratio", 1.0*tp/positive_number)
                print("false positive ratio", 1.0*fp/negative_number)
                print("false negative ratio", 1.0*fn/positive_number)
                
                                
                # compute tpr, given fpr
                fpr_origin, tpr_origin, thresholds = roc_curve(real_data_test[:,-1], probs_real, pos_label=1)
                plt.figure()
                plt.plot(fpr_origin, tpr_origin, color='darkorange', label='ROC curve (area = %0.3f)' % (auc_real))
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                plt.title('Original dataset - Receiver operating characteristic')
                plt.legend(loc='lower right')
                plt.show()
                
                index = np.argmin(abs(fpr_origin-0.0001))
                print(fpr_origin[index], tpr_origin[index], thresholds[index])
                print('---')
                
                # Train the generated data
                
                print('Training classifier with synthetic data.')
                if(classifier=='LogisticRegression'):
                    lg_gen = LogisticRegression(random_state=0, solver='saga').fit(gen_data[:,:-1], gen_data[:,-1]) #logistic regression
                elif(classifier=='MLPClassifier'):
                    lg_gen = MLPClassifier(activation = 'tanh',hidden_layer_sizes = (18,18,18)).fit(gen_data[:,:-1], gen_data[:,-1]) #decision tree
                elif(classifier=='AdaBoostClassifier'):
                    lg_gen = AdaBoostClassifier(base_estimator = LogisticRegression(),n_estimators = 200).fit(gen_data[:,:-1], gen_data[:,-1]) #adaboost
                elif(classifier=='BaggingClassifier'):
                    lg_gen = BaggingClassifier(base_estimator = LogisticRegression(),n_estimators = 100).fit(gen_data[:,:-1], gen_data[:,-1]) #random forest
                print('Testing classifier ... ')
                probs_gen = lg_gen.predict_proba(real_data_test[:,:-1])
                probs_gen = probs_gen[:, 1]
                label_gen = lg_gen.predict(real_data_test[:,:-1])
                #label_gen = np.where(lg_real.predict_proba(real_data_test[:,:-1]) >= threshold, 0, 1)[:,0]
                auc_gen = roc_auc_score(real_data_test[:,-1], probs_gen)
                print("auc_gen", auc_gen)
                confusion_matrix_gen = confusion_matrix(real_data_test[:,-1], label_gen)
                tn, fp, fn, tp = confusion_matrix_gen.ravel()
                print('tn = %f, fp = %f, fn = %f, tp = %f' % (tn, fp, fn, tp))
                print("true positive ratio", 1.0*tp/positive_number)
                print("false positive ratio", 1.0*fp/negative_number)
                print("false negative ratio", 1.0*fn/positive_number)
                
                # compute tpr, given fpr
                fpr, tpr, thresholds = roc_curve(real_data_test[:,-1], probs_gen, pos_label=1)
                plt.figure()
                plt.plot(fpr_origin, tpr_origin, color='darkorange', label='ROC curve - Original dataset (area = %0.3f)' % (auc_real))
                plt.plot(fpr, tpr, color='darkblue', label='ROC curve - Dist-GAN DP (area = %0.3f)' % (auc_gen))
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                plt.title('Dist-GAN - Receiver operating characteristic')
                plt.legend(loc='lower right')
                plt.savefig(self.ckpt_dir + '/dpdistgan_epsilon_' + str(self.eps)+ '_' + classifier + '.png')
                plt.show()
                
                # compute tp with respect to fp
                '''
                x = min(abs(fpr-0.0001))
                if x+0.01 in list(fpr):
                    index = list(fpr).index(x+0.0001)
                else:
                    index = list(fpr).index(0.0001-x)
                print(len(fp))
                print(len(tp))
                print(fpr[index], tpr[index], thresholds[index])
                '''
                
                index = np.argmin(abs(fpr-0.0001))
                print(fpr[index], tpr[index], thresholds[index])

    def get_min_max(self):
        from numpy import genfromtxt
        train_file = './data/cervical_cancer2/multi_impute_orignaldata_train.csv'
        X = genfromtxt(train_file, delimiter=',')
        min_max_list = []
        for column in range(X.shape[1]):
            min_f = min(X[:,column])
            max_f = max(X[:,column])
            min_max_list.append([min_f, max_f])
        min_max_array = np.array(min_max_list)
        print(min_max_array.T)
        np.savetxt("./data/cervical_cancer2/CervicalCancerData_min_max_train.csv", min_max_array.T, delimiter=",")
