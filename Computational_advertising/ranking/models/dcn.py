#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 18:40:18 2019

@author: zhangkai
"""

import tensorflow as tf
from .fm import fm
import numpy as np

def dcn(embeddings, cross_layer_num, field_size):
    # input: embeddings should be batch * filed * embedding_dim, cross_layer_num is number of corss times
    # output: cross_network_out
    # conside if we can put every output of dcn to dnn, and dnn to dcn
    
    weights = dict()
    glorot = np.sqrt(2.0/(field_size + field_size))
    
    for i in range(cross_layer_num):
        weights["cross_layer_%d" % i] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(field_size,1)),
            dtype=np.float32)
        weights["cross_bias_%d" % i] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(field_size,1)),
            dtype=np.float32)  
    

    batch_x0 = embeddings
    x_l = batch_x0
    
    for l in range(cross_layer_num):
        x_l = tf.tensordot(tf.matmul(batch_x0, x_l, transpose_b=True),
                            weights["cross_layer_%d" % l],1) + weights["cross_bias_%d" % l] + x_l
    
    cross_network_out = tf.reshape(x_l, (-1, field_size))
    
    return cross_network_out  
