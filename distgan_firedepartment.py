import os
import numpy as np
from modules.dataset import Dataset
from distgan import DISTGAN
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


if __name__ == '__main__':
    
    out_dir = 'output/'
    db_name = 'fire_department'
    data_source = './data/fire/preprocess_2.csv'
    
    #Parser
    parser = argparse.ArgumentParser('')
    parser.add_argument('--train', type=str, default='0')
    parser.add_argument('--steps', type=str, default='100000')
    args = parser.parse_args()
    is_train = int(args.train)
    
    # network and loss types
    model     = 'distgan' 
    nnet_type = 'fcnet' # 'fcnet'
    loss_type = 'log'   # 'log'
    
    # improved contraints
    gngan = 0    #0: no gradient matching and neighbor embedding
                 #1: + neighbor embedding
                 #2: + gradient matching
                 #3: + neighbor embedding + gradient matching    
    
    ssgan = 0    #0: no self-supervised training
                 #1: + self-supervised for discriminator
                 #2: + self-supervised for discriminator + g loss
                 #3: + self-supervised for discriminator (adversarial) + g loss
                     
    '''
    0: do not convert into categorical
    1: convert to categorical but does not use softmax
    2: convert to categorical but does use softmax
    3: convert to categorical but does use gumble-softmax
    '''
    categorical_softmax_use = 3
    
    '''
    model parameters
    '''
    if categorical_softmax_use == 0:
        noise_dim    = 93  #latent dim / feature dim (original 36, categorical 55)
        feature_dim  = 93  #feture dim, set your self as in the paper
    else:
        noise_dim    = 195  #latent dim / feature dim (original 36, categorical 55)
        feature_dim  = 195 #feture dim, set your self as in the paper       
    
    '''
    differential privacy parameters
    '''
    C     = 0    # clipping threshold of gradient.
    eps   = 100000.0    # the eps of differential privacy.
    delta = 1e-5   # the delta of differential privacy.
    
    regc  = 2.5e-5 # regualrization of model parameters.
    
    lr    = 1e-4   # 2e-4: Dist-GAN, 1e-4: DP Dist-GAN.
        
    n_steps  = int(args.steps) #number of iterations
    
    #ssgan
    if nnet_type == 'fcnet':
        
        # ssagan
        lambda_d  = 0.0  #1.0 discriminator
        lambda_g  = 0.0  #0.1 generator

        # network
        df_dim = feature_dim
        gf_dim = noise_dim
        ef_dim = feature_dim
        beta1  = 0.5
        beta2  = 0.9
            
    lambda_p  = 0.5
    lambda_r  = 0.0
    
    # [Impotant]
    # lambda_w = sqrt(d/D) as in the paper, if you change the network 
    #  architecture: (d: data noise dim, D: feature dim)
    lambda_w  = 0 #np.sqrt(noise_dim * 1.0/feature_dim)
    
    if C == 0:
        ext_name = '%d_lr_%f_2dups_max_0.softmax_categorical_%d_tau_1.0' % (n_steps, lr, categorical_softmax_use)
    else:
        ext_name = 'C_%d_eps_%f_delta_%f_dp_%d_lr_%f_dups10_softmax_categorical_%d_tau_0.1' % (C, eps, delta, n_steps, lr, categorical_softmax_use)
        #ext_name = 'C_%d_eps_%f_delta_%f_dp_%d_lr_%f_dups10_softmax_categorical_%d_tau_1.0_hard_True' % (C, eps, delta, n_steps, lr, categorical_softmax_use)
        #ext_name = 'C_%d_eps_%f_delta_%f_dp_%d_lr_%f_dups10_softmax_categorical_%d_cervical_cancer2' % (C, eps, delta, n_steps, lr, categorical_softmax_use)
    
    #output dir
    out_dir = os.path.join(out_dir, db_name + '_' + model + '_' \
                                          + nnet_type + '_' \
                                          + loss_type + '_' \
                                          + ext_name, db_name)

    #out_dir = os.path.join(out_dir, ext_name)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # setup dataset
    dataset = Dataset(name=db_name, source=data_source, categorical_softmax_use = categorical_softmax_use, batch_size=64)
        
    # setup gan model and train
    distgan = DISTGAN(model=model, \
                              is_train = is_train, \
                              loss_type = loss_type, \
                              lambda_p=lambda_p, \
                              lambda_r=lambda_r, \
                              lambda_w=lambda_w, \
                              lambda_d=lambda_d, \
                              lambda_g=lambda_g, \
                              C   = C, \
                              eps = eps, \
                              delta = delta, \
                              regc = regc,\
                              lr = lr, \
                              noise_dim = noise_dim, \
                              beta1 = beta1, \
                              beta2 = beta2, \
                              nnet_type = nnet_type, \
                              df_dim = df_dim, \
                              gf_dim = gf_dim, \
                              ef_dim = ef_dim, \
                              gngan  = gngan,  \
                              ssgan  = ssgan,  \
                              dataset=dataset, \
                              n_steps = n_steps, \
                              out_dir=out_dir, batch_size=64)
    
    if is_train == 0:
        distgan.train()
    elif is_train == 1:
        distgan.generate()
    elif is_train == 2:
        pass
    elif is_train == 3:
        pass
    elif is_train == 4:
        distgan.get_min_max()
        distgan.eval_acc(classifier='LogisticRegression')
        '''
        classifier: LogisticRegression, DecisionTreeClassifier, AdaBoostClassifier, RandomForestClassifier
        '''
    elif is_train == 5:
        pass
