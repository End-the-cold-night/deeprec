#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:05:55 2019

@author: zhangkai

Guo H, Tang R, Ye Y, et al. DeepFM: An End-to-End Wide & Deep Learning Framework for CTR Prediction[J]. arXiv preprint arXiv:1804.04950, 2018.
"""

import tensorflow as tf
from .fm import fm
import numpy as np

def deepfm(embeddings, field_size, embedding_size, deep_layers=[128, 64, 32], dropout_keep_deep=[0.5, 0.5, 0.5, 0.5], deep_layers_activation=tf.nn.relu, use_fm=1, use_deep=1):
    """
    embeddings: embeddings should be batch * filed * embedding_dim
    field_size:
    embedding_size:
    deep_layers: how many layers has in deep part
    dropout_keep_deep: dropout in every layers, this dim just one large than deep_layers, for input dropout
    deep_layer_activation
    use_deep
    use_fm
    """
    
    ###### fm part  #######
    fm_part = fm(embeddings)
    
    ######  deep part  ###### 
    ## weights
    weights = dict()
    glorot = np.sqrt(2.0 / (field_size * embedding_size + deep_layers[0]))
    weights["layer_%d" % 0] = tf.Variable(
        np.random.normal(loc=0, scale=glorot, size=(field_size * embedding_size, deep_layers[0])),
        dtype=np.float32)  
    weights["bias_%d" % 0] = tf.Variable(
        np.random.normal(loc=0, scale=glorot, size=(1, deep_layers[0])),
        dtype=np.float32)  # 1 * layer[i]    
    
    for i in range(1,len(deep_layers)):    
        glorot = np.sqrt(2.0 / (deep_layers[i - 1] + deep_layers[i]))
        weights["layer_%d" % i] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(deep_layers[i - 1], deep_layers[i])),
            dtype=np.float32) 
        weights["bias_%d" % i] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(1, deep_layers[i])),
            dtype=np.float32) 

    ## models
    deep_part = tf.reshape(embeddings,shape=[-1,field_size * embedding_size])
    deep_part = tf.nn.dropout(deep_part,dropout_keep_deep[0])

    for i in range(0,len(deep_layers)):
        deep_part = tf.add(tf.matmul(deep_part,weights["layer_%d" %i]), weights["bias_%d"%i])
        deep_part = deep_layers_activation(deep_part)
        deep_part = tf.nn.dropout(deep_part,dropout_keep_deep[i+1])    
        

    if use_fm and use_deep:
        concat_input = tf.concat([fm_part, deep_part], axis=1)
    elif use_fm:
        concat_input = fm_part
    elif use_deep:
        concat_input = deep_part        
        
    return concat_input
