import numpy as np
import random
import math
from skimage import io, transform
from skimage.transform import resize
import tensorflow as tf
import copy
import scipy
#from attacks import cw
#from attacks import fgm

def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        path_dir, _ = os.path.split(path)
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir)

def imwrite(image, path):
    """ save an [-1.0, 1.0] image """
    #image = (image - 0.5) * 2
    if image.ndim == 3 and image.shape[2] == 1:  # for gray image
        image = np.array(image, copy=True)
        image.shape = image.shape[0:2]
    return io.imsave(path, image)
    #return scipy.misc.imsave(path, image)
     
def immerge_row_col(N):
    c = int(np.floor(np.sqrt(N)))
    for v in range(c,N):
        if N % v == 0:
            c = v
            break
    r = N / c
    return r, c
    
def immerge(images, row, col):
    """
    merge images into an image with (row * h) * (col * w)
    @images: is in shape of N * H * W(* C=1 or 3)
    """
    row = int(row)
    col = int(col)
    h, w = images.shape[1], images.shape[2]
    if images.ndim == 4:
        img = np.zeros((h * row, w * col, images.shape[3]))
    elif images.ndim == 3:
        img = np.zeros((h * row, w * col))
    for idx, image in enumerate(images):
        i = idx % col
        j = idx // col
        img[j * h:j * h + h, i * w:i * w + w, ...] = image
    return img


def tf_rotate(X, data_shape):
    nimgs = X.get_shape().as_list()[0]
    angle = math.pi / 180
    n = nimgs
    Xtmp  = tf.reshape(X,[nimgs,data_shape[0],data_shape[1],data_shape[2]])

    # rotate images with different angles
    v_0  = tf.constant([[0]])
    X_0  = Xtmp[:n,:,:,:]
    l_0  = tf.tile(v_0, [n,1])
    
    v_1  = tf.constant([[1]])
    X_1  = tf.contrib.image.rotate(X_0, 90 * angle)
    l_1  = tf.tile(v_1, [n,1])
    
    v_2  = tf.constant([[2]])
    X_2  = tf.contrib.image.rotate(X_0, 180 * angle)
    l_2  = tf.tile(v_2, [n,1])
    
    v_3  = tf.constant([[3]])
    X_3  = tf.contrib.image.rotate(X_0, 270 * angle)
    l_3  = tf.tile(v_3, [n,1])          
    
    Xarg = tf.concat([X_0, X_1, X_2, X_3], axis=0)
    larg = tf.concat([l_0, l_1, l_2, l_3], axis=0)
    
    return Xarg, larg

def tf_argument_image_rotation(X, data_shape, ridx=None):
    
    nimgs = X.get_shape().as_list()[0]
    angle = math.pi / 180
    
    Xtmp  = tf.reshape(X,[nimgs,data_shape[0],data_shape[1],data_shape[2]])
    n     = nimgs
            
    # argument a quarter only
    v_0  = tf.constant([[1., 0., 0., 0.]])
    X_0  = Xtmp[:n,:,:,:]
    l_0  = tf.tile(v_0, [n,1])
    
    v_1  = tf.constant([[0., 1., 0., 0.]])
    X_1  = tf.contrib.image.rotate(X_0, 90 * angle)
    l_1  = tf.tile(v_1, [n,1])
    
    v_2  = tf.constant([[0., 0., 1., 0.]])
    X_2  = tf.contrib.image.rotate(X_0, 180 * angle)
    l_2  = tf.tile(v_2, [n,1])
    
    v_3  = tf.constant([[0., 0., 0., 1.]])
    X_3  = tf.contrib.image.rotate(X_0, 270 * angle)
    l_3  = tf.tile(v_3, [n,1])

    Xarg = tf.concat([X_0, X_1, X_2, X_3], axis=0)
    larg = tf.concat([l_0, l_1, l_2, l_3], axis=0)
    
    Xarg = tf.reshape(Xarg,[-1,data_shape[0] * data_shape[1] * data_shape[2]])
    
    if ridx is None:
        ridx = tf.range(0,nimgs*4,1)
        ridx = tf.expand_dims(tf.random_shuffle(ridx),axis=1)[0:nimgs,:]
        
    Xarg = tf.gather_nd(Xarg, ridx)
    larg = tf.gather_nd(larg, ridx)
        
    return Xarg, larg, ridx

def tf_argument_image_rotation_plus_fake(X, data_shape, ridx=None):
    
    nimgs = X.get_shape().as_list()[0]
    angle = math.pi / 180
    
    Xtmp  = tf.reshape(X,[nimgs,data_shape[0],data_shape[1],data_shape[2]])
    n     = nimgs
            
    # argument a quarter only
    v_0  = tf.constant([[1., 0., 0., 0., 0.]])
    X_0  = Xtmp[:n,:,:,:]
    l_0  = tf.tile(v_0, [n,1])
    
    v_1  = tf.constant([[0., 1., 0., 0., 0.]])
    X_1  = tf.contrib.image.rotate(X_0, 90 * angle)
    l_1  = tf.tile(v_1, [n,1])
    
    v_2  = tf.constant([[0., 0., 1., 0., 0.]])
    X_2  = tf.contrib.image.rotate(X_0, 180 * angle)
    l_2  = tf.tile(v_2, [n,1])
    
    v_3  = tf.constant([[0., 0., 0., 1., 0.]])
    X_3  = tf.contrib.image.rotate(X_0, 270 * angle)
    l_3  = tf.tile(v_3, [n,1])

    Xarg = tf.concat([X_0, X_1, X_2, X_3], axis=0)
    larg = tf.concat([l_0, l_1, l_2, l_3], axis=0)
    
    Xarg = tf.reshape(Xarg,[-1,data_shape[0] * data_shape[1] * data_shape[2]])
    
    if ridx is None:
        ridx = tf.range(0,nimgs*4,1)
        ridx = tf.expand_dims(tf.random_shuffle(ridx),axis=1)[0:nimgs,:]
        #ridx  = tf.random_uniform([nimgs,1], 0, nimgs*4, tf.int64)
        
    Xarg = tf.gather_nd(Xarg, ridx)
    larg = tf.gather_nd(larg, ridx)
        
    return Xarg, larg, ridx

def tf_argument_image_rotation_and_fake_mix(X, X_f, data_shape, ridx=None): #iccv 2019 only good with tensorflow 1.1.0
    
    nimgs = X.get_shape().as_list()[0]
    angle = math.pi / 180
    
    Xtmp       = tf.reshape(X,  [nimgs,data_shape[0],data_shape[1],data_shape[2]])
    Xtmp_fake  = tf.reshape(X_f,[nimgs,data_shape[0],data_shape[1],data_shape[2]])
    n          = nimgs
            
    # argument a quarter only
    v_0  = tf.constant([[1., 0., 0., 0., 0.]])
    X_0  = Xtmp[:n,:,:,:]
    l_0  = tf.tile(v_0, [n,1])
    
    v_1  = tf.constant([[0., 1., 0., 0., 0.]])
    X_1  = tf.contrib.image.rotate(X_0, 90 * angle)
    l_1  = tf.tile(v_1, [n,1])
    
    v_2  = tf.constant([[0., 0., 1., 0., 0.]])
    X_2  = tf.contrib.image.rotate(X_0, 180 * angle)
    l_2  = tf.tile(v_2, [n,1])
    
    v_3  = tf.constant([[0., 0., 0., 1., 0.]])
    X_3  = tf.contrib.image.rotate(X_0, 270 * angle)
    l_3  = tf.tile(v_3, [n,1])

    v_4  = tf.constant([[0., 0., 0., 0., 1.]])
    X_4  = Xtmp_fake[:n,:,:,:]
    l_4  = tf.tile(v_4, [n,1])

    Xarg = tf.concat([X_0, X_1, X_2, X_3, X_4], axis=0)
    larg = tf.concat([l_0, l_1, l_2, l_3, l_4], axis=0)
    
    Xarg = tf.reshape(Xarg,[-1,data_shape[0] * data_shape[1] * data_shape[2]])
    
    if ridx is None:
        ridx = tf.range(0,nimgs*5,1)
        ridx = tf.expand_dims(tf.random_shuffle(ridx),axis=1)[0:nimgs,:]
        #ridx  = tf.random_uniform([nimgs,1], 0, nimgs*5, tf.int64)
        
    Xarg = tf.gather_nd(Xarg, ridx)
    larg = tf.gather_nd(larg, ridx)
        
    return Xarg, larg, ridx

def tf_argument_image_rotation_and_fake_mix2(X, X_f, data_shape):
    
    nimgs = X.get_shape().as_list()[0]
    angle = math.pi / 180
    
    Xtmp       = tf.reshape(X,  [nimgs,data_shape[0],data_shape[1],data_shape[2]])
    Xtmp_fake  = tf.reshape(X_f,[nimgs,data_shape[0],data_shape[1],data_shape[2]])
    n          = nimgs
            
    # argument a quarter only
    v_0  = tf.constant([[1., 0., 0., 0., 0.]])
    X_0  = Xtmp[:n,:,:,:]
    l_0  = tf.tile(v_0, [n,1])
    
    v_1  = tf.constant([[0., 1., 0., 0., 0.]])
    X_1  = tf.contrib.image.rotate(X_0, 90 * angle)
    l_1  = tf.tile(v_1, [n,1])
    
    v_2  = tf.constant([[0., 0., 1., 0., 0.]])
    X_2  = tf.contrib.image.rotate(X_0, 180 * angle)
    l_2  = tf.tile(v_2, [n,1])
    
    v_3  = tf.constant([[0., 0., 0., 1., 0.]])
    X_3  = tf.contrib.image.rotate(X_0, 270 * angle)
    l_3  = tf.tile(v_3, [n,1])

    v_4  = tf.constant([[0., 0., 0., 0., 1.]])
    X_4  = Xtmp_fake[:n,:,:,:]
    l_4  = tf.tile(v_4, [n,1])

    Xarg_real = tf.concat([X_0, X_1, X_2, X_3], axis=0)
    larg_real = tf.concat([l_0, l_1, l_2, l_3], axis=0)
    
    Xarg_fake = X_4
    larg_fake = l_4
    
    Xarg_real = tf.reshape(Xarg_real,[-1,data_shape[0] * data_shape[1] * data_shape[2]])
    Xarg_fake = tf.reshape(Xarg_fake,[-1,data_shape[0] * data_shape[1] * data_shape[2]])
    
    # random to pickup samples for real and fake
    ridx_real = tf.range(0,nimgs*4,1)
    ridx_real = tf.expand_dims(tf.random_shuffle(ridx_real),axis=1)[0:int(nimgs*4/5)+1,:]
    ridx_fake = tf.range(0,nimgs,1)
    ridx_fake = tf.expand_dims(tf.random_shuffle(ridx_fake),axis=1)[0:int(nimgs/5)+1,:]
        
    Xarg_real = tf.gather_nd(Xarg_real, ridx_real)
    larg_real = tf.gather_nd(larg_real, ridx_real)
    
    Xarg_fake = tf.gather_nd(Xarg_fake, ridx_fake)
    larg_fake = tf.gather_nd(larg_fake, ridx_fake)
    
    Xarg = tf.concat([Xarg_real, Xarg_fake], axis=0)
    larg = tf.concat([larg_real, larg_fake], axis=0)
    
    # random to select nimgs samples
    nimgs_arg = Xarg.get_shape().as_list()[0]
    ridx      = tf.range(0,nimgs_arg,1)
    ridx      = tf.expand_dims(tf.random_shuffle(ridx),axis=1)[0:nimgs,:]
    
    Xarg = tf.gather_nd(Xarg, ridx)
    larg = tf.gather_nd(larg, ridx)
        
    return Xarg, larg, ridx

def diversity_score(ridx, nimgs, loss_type='cluster'):
    '''
    @loss_type: cluster or entropy
    '''
    eps = 0.001
    if loss_type == 'cluster':
        ridx_dv     = tf.floormod(ridx, nimgs)       # return remainder of division
        group_dv, _ = tf.unique(tf.squeeze(ridx_dv)) # unique to have cluster id
        ng          = tf.size(group_dv)              # get number of clusters
        return ng
    elif loss_type == 'shannon':
        # batch 64, [max, min] = [0.28413569927215576, 0.17873544991016388]
        ridx_dv     = tf.floormod(ridx, nimgs)       # return remainder of division
        hist        = tf.bincount(ridx_dv)           # compute occurences of each value
        hist        = tf.cast(hist,tf.float32) / 4.0 # because of 4 rotations
        en          = - tf.reduce_sum(tf.multiply(hist, tf.log(hist + eps))) / nimgs
        # normaliztion to close (0, 1)
        en          = (en - 0.17) * 7.0
        return en
    
def tf_argument_image_rotation_diversity(X, data_shape, loss_type, ridx=None):
    
    nimgs = X.get_shape().as_list()[0]
    angle = math.pi / 180
    
    Xtmp  = tf.reshape(X,[nimgs,data_shape[0],data_shape[1],data_shape[2]])
    n     = nimgs
            
    # argument a quarter only
    v_0  = tf.constant([[1., 0., 0., 0.]])
    X_0  = Xtmp[:n,:,:,:]
        
    v_1  = tf.constant([[0., 1., 0., 0.]])
    X_1  = tf.contrib.image.rotate(X_0, 90 * angle)
    
    v_2  = tf.constant([[0., 0., 1., 0.]])
    X_2  = tf.contrib.image.rotate(X_0, 180 * angle)
        
    v_3  = tf.constant([[0., 0., 0., 1.]])
    X_3  = tf.contrib.image.rotate(X_0, 270 * angle)
    
    Xarg = tf.concat([X_0, X_1, X_2, X_3], axis=0)
    Xarg = tf.reshape(Xarg,[-1,data_shape[0] * data_shape[1] * data_shape[2]])
    
    if ridx is None:
        ridx = tf.range(0,nimgs*4,1)
        ridx = tf.expand_dims(tf.random_shuffle(ridx),axis=1)[0:nimgs,:]
        
    div_score = diversity_score(ridx, nimgs, loss_type)
    
    Xarg = tf.gather_nd(Xarg, ridx)
            
    return Xarg, ridx, div_score


#Hung's function
def tf_argument_image_rotation_adversarial(model, X, data_shape, ridx=None):
    
    nimgs = X.get_shape().as_list()[0]
    angle = math.pi / 180
    
    Xtmp  = tf.reshape(X,[nimgs,data_shape[0],data_shape[1],data_shape[2]])
    n     = nimgs
            
    # argument a quarter only
    v_0  = tf.constant([[1., 0., 0., 0.]])
    X_0  = Xtmp[:n,:,:,:]
    l_0  = tf.tile(v_0, [n,1])
    
    v_1  = tf.constant([[0., 1., 0., 0.]])
    X_1  = tf.contrib.image.rotate(X_0, 90 * angle)
    l_1  = tf.tile(v_1, [n,1])
    
    v_2  = tf.constant([[0., 0., 1., 0.]])
    X_2  = tf.contrib.image.rotate(X_0, 180 * angle)
    l_2  = tf.tile(v_2, [n,1])
    
    v_3  = tf.constant([[0., 0., 0., 1.]])
    X_3  = tf.contrib.image.rotate(X_0, 270 * angle)
    l_3  = tf.tile(v_3, [n,1])

    Xarg = tf.concat([X_0, X_1, X_2, X_3], axis=0)
    larg = tf.concat([l_0, l_1, l_2, l_3], axis=0)
    
    Xarg = tf.reshape(Xarg,[-1,data_shape[0] * data_shape[1] * data_shape[2]])
    
    if ridx is None:
        ridx = tf.range(0,nimgs*4,1)
        ridx = tf.expand_dims(tf.random_shuffle(ridx),axis=1)[0:nimgs,:]
    print(Xarg)    
    Xarg = tf.gather_nd(Xarg, ridx)
    larg = tf.gather_nd(larg, ridx)

    X_avd = fgm(model=model, x=Xarg, x_shape=data_shape)
    # X_avd = tf.identity(Xarg)
    # print(X)
    # print(Xarg)
    # print(X_avd)
    # exit()
    # save generated images
        
    return Xarg, X_avd, larg, ridx

def tf_argument_image_rotation_adversarial_2(model, X, X_f, data_shape, ridx=None):
    
    nimgs = X.get_shape().as_list()[0]
    angle = math.pi / 180
    
    Xtmp  = tf.reshape(X,[nimgs,data_shape[0],data_shape[1],data_shape[2]])
    Xtmp_fake  = tf.reshape(X_f,[nimgs,data_shape[0],data_shape[1],data_shape[2]])
    n     = nimgs
            
    # argument a quarter only
    v_0  = tf.constant([[1., 0., 0., 0., 0.]])
    X_0  = Xtmp[:n,:,:,:]
    l_0  = tf.tile(v_0, [n,1])
    
    v_1  = tf.constant([[0., 1., 0., 0., 0.]])
    X_1  = tf.contrib.image.rotate(X_0, 90 * angle)
    l_1  = tf.tile(v_1, [n,1])
    
    v_2  = tf.constant([[0., 0., 1., 0., 0.]])
    X_2  = tf.contrib.image.rotate(X_0, 180 * angle)
    l_2  = tf.tile(v_2, [n,1])
    
    v_3  = tf.constant([[0., 0., 0., 1., 0.]])
    X_3  = tf.contrib.image.rotate(X_0, 270 * angle)
    l_3  = tf.tile(v_3, [n,1])

    v_4  = tf.constant([[0., 0., 0., 0., 1.]])
    X_4  = Xtmp_fake[:n,:,:,:]
    l_4  = tf.tile(v_4, [n,1])

    Xarg = tf.concat([X_0, X_1, X_2, X_3, X_4], axis=0)
    larg = tf.concat([l_0, l_1, l_2, l_3, l_4], axis=0)

    Xarg = tf.concat([X_0, X_1, X_2, X_3, X_4], axis=0)
    larg = tf.concat([l_0, l_1, l_2, l_3, l_4], axis=0)
    
    Xarg = tf.reshape(Xarg,[-1,data_shape[0] * data_shape[1] * data_shape[2]])
    
    if ridx is None:
        ridx = tf.range(0,nimgs*5,1)
        ridx = tf.expand_dims(tf.random_shuffle(ridx),axis=1)[0:nimgs,:]
    print(Xarg)    
    Xarg = tf.gather_nd(Xarg, ridx)
    larg = tf.gather_nd(larg, ridx)

    X_avd = fgm(model=model, x=Xarg, x_shape=data_shape)
    # X_avd = tf.identity(Xarg)
    # print(X)
    # print(Xarg)
    # print(X_avd)
    # exit()
    # save generated images
        
    return Xarg, X_avd, larg, ridx


