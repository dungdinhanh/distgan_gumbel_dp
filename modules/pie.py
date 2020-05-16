import numpy as np 
import tensorflow as tf 

def get_shape(x):
	return x.get_shape().as_list()
	
	
def pairwise_distance(x, y):
    s = get_shape(x)
    size_x = tf.shape(x)[0]
    size_y = tf.shape(y)[0]
    xx = tf.expand_dims(x, -1)
    xx = tf.tile(xx, [1, 1, size_y])

    yy = tf.expand_dims(y, -1)
    yy = tf.tile(yy, [1, 1, size_x])
    yy = tf.transpose(yy, perm=[2, 1, 0])

    diff = xx - yy
    square_diff = tf.square(diff)

    square_dist = tf.reduce_sum(square_diff, 1)

    square_dist = square_dist / np.prod(s[1:]) 
       
    return square_dist

def get_variance(x):
	x_ = tf.reshape(x, [-1])
	u_ = tf.reduce_mean(x)
	v_ = tf.sqrt(tf.reduce_mean(tf.square(x_ - u_)))

	return v_

def get_variance_in_row(x): 
    m,n = get_shape(x)
    u_ = tf.reduce_mean(x, axis=1)
    u_ = tf.expand_dims(u_, axis=1)
    u_ = tf.tile(u_, [1, n])
    d = x - u_
    d = tf.square(d)
    v = tf.reduce_mean(d, axis=1)
    v = tf.sqrt(v)
    return v 

def get_dist_table(x, dist, symmetric, alpha):
    batch_size = get_shape(x)[0]
    P = pairwise_distance(x, x)
    v_ = get_variance(P)

    if dist == 'gauss':
        P = P / v_
        P = tf.exp(-P)
    elif dist == 'tdis':
        P = tf.pow(1. + P / v_ / alpha, -1.)

    toset = tf.constant(0., shape=[batch_size], dtype=tf.float32)
    P = tf.matrix_set_diag(P, toset) 

    if symmetric == True:
        m = tf.reduce_sum(P)
        P = P / m 
    else:
        m = tf.reduce_sum(P, axis=1)
        m = tf.tile(tf.expand_dims(m, axis=1), [1, batch_size])
        P = tf.div(P, m)
        P = 0.5 * (P + tf.transpose(P))
        P = P / batch_size
    return P
    
    
def get_dist_table_novariance(x, dist, symmetric, alpha):
    batch_size = get_shape(x)[0]
    P = pairwise_distance(x, x)
    
    if dist == 'gauss':
        P = tf.exp(-P)
    elif dist == 'tdis':
        P = tf.pow(1. + P, -1.)

    toset = tf.constant(0., shape=[batch_size], dtype=tf.float32)
    P = tf.matrix_set_diag(P, toset) 

    if symmetric == True:
        m = tf.reduce_sum(P)
        P = P / m 
    else:
        m = tf.reduce_sum(P, axis=1)
        m = tf.tile(tf.expand_dims(m, axis=1), [1, batch_size])
        P = tf.div(P, m)
        P = 0.5 * (P + tf.transpose(P))
        P = P / batch_size
    return P

def get_KL_loss(P, Q):
	esilon = tf.constant(1e-7, dtype=tf.float32)

	kl = tf.multiply(P, tf.log(P+esilon) - tf.log(Q+esilon))
	return tf.reduce_sum(kl)


if __name__ == '__main__':
    a = np.random.uniform(low=0.1, high=0.9, size=(64,128))
    a = np.concatenate([a, a[0,:] + 0.9 * np.ones(shape=(1,128))], axis=0)
    a = tf.constant(a, dtype=tf.float32)
    p = pairwise_distance(a, a)

    b = np.random.uniform(low=0.1, high=0.9, size=(64,128))
    b = np.concatenate([b, b[0,:] + 0.01 * np.ones(shape=(1,128))], axis=0)
    b = tf.constant(b, dtype=tf.float32)
    q = pairwise_distance(b, b)

    v = get_variance(p)
    p1 = p/v 
    p2 = tf.exp(-p1)
    p3 = p2 / tf.reduce_sum(p2)

    p4 = tf.pow(1. + p, -1)
    p5 = p4 / tf.reduce_sum(p4)

    u = get_variance(q)
    q1 = q / u 
    q2 = tf.exp(-q1)
    q3 = q2 / tf.reduce_sum(q2)

    q4 = tf.pow(1. + q, -1)
    q5 = q4 / tf.reduce_sum(q4)

    kl1 = get_KL_loss(p3, q3)
    kl2 = get_KL_loss(p5, q5)

    p = get_dist_table(a, dist='tdis', symmetric=True, alpha=1.)
    q = get_dist_table(b, dist='tdis', symmetric=True, alpha=1.)
    kl3 = get_KL_loss(p, q)

    with tf.Session() as sess: 
        print(sess.run(v))
        print(sess.run(p1))
        print('---')
        print(sess.run(p2))
        print('---')
        print(sess.run(p3))
        print('---')
        print(sess.run(p4))
        print('---')
        print(sess.run(p5))
        print('---')
        print(sess.run(kl1))
        print(sess.run(kl2))
        print(sess.run(kl3))



