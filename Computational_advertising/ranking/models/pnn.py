import tensorflow as tf
from .fm import fm
import numpy as np

def pnn(embeddings, deep_init_size, field_size, embedding_size, use_inner):
    # input: embeddings should be batch * filed * embedding_dim
    # deep_init_size is the size of product layer 
    weights = dict()    
    if use_inner:
        weights['product-quadratic-inner'] = tf.Variable(tf.random_normal([deep_init_size,field_size],0.0,0.01))  
    else:
        weights['product-quadratic-outer'] = tf.Variable(
            tf.random_normal([deep_init_size, embedding_size,embedding_size], 0.0, 0.01))
    weights['product-linear'] = tf.Variable(tf.random_normal([deep_init_size,field_size,embedding_size],0.0,0.01))  
    weights['product-bias'] = tf.Variable(tf.random_normal([deep_init_size,],0,0,1.0))

    # Linear Singal, weights['product-linear'] is D1  * F * K
    linear_output = []
    for i in range(deep_init_size):
        linear_output.append(tf.reshape(
            tf.reduce_sum(tf.multiply(embeddings,weights['product-linear'][i]),axis=[1,2]),shape=(-1,1)))# N * 1
    lz = tf.concat(linear_output,axis=1) # N * init_deep_size


    # Quardatic Singa
    quadratic_output = []
    if use_inner:
        # INPP weights['product-quadratic-inner'][i],(1,-1,1))
        for i in range(deep_init_size):
            theta = tf.multiply(embeddings,tf.reshape(weights['product-quadratic-inner'][i],(1,-1,1))) # N * F * K
            quadratic_output.append(tf.reshape(tf.norm(tf.reduce_sum(theta,axis=1),axis=1),shape=(-1,1))) # N * 1

    else:
        embedding_sum = tf.reduce_sum(embeddings,axis=1)  # 
        p = tf.matmul(tf.expand_dims(embedding_sum,2),tf.expand_dims(embedding_sum,1)) # N * K * K
        for i in range(deep_init_size):
            theta = tf.multiply(p,tf.expand_dims(weights['product-quadratic-outer'][i],0)) # N * K * K
            quadratic_output.append(tf.reshape(tf.reduce_sum(theta,axis=[1,2]),shape=(-1,1))) # N * 1
    lp = tf.concat(quadratic_output,axis=1) # N * init_deep_size

    y_deep = tf.nn.relu(tf.add(tf.add(lz, lp), weights['product-bias']))

    return y_deep   #  batch * init_deep_size

