#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 22:05:16 2019

@author: zhangkai
"""
import numpy as np
import tensorflow as tf
def cin(embeddings, field_size, embedding_size, deep_layers_activation=tf.nn.relu, num_layers=[64, 64, 32],  direct=False, bias=False):
    # embeddings 
    #
    #
    #
    
    hidden_nn_layers = []
    field_sizes = []
    final_len = 0
    embeddings = tf.reshape(embeddings, shape=[-1, int(field_size), embedding_size])
    field_sizes.append(int(field_size))
    hidden_nn_layers.append(embeddings)
    final_result = []
    split_tensor0 = tf.split(hidden_nn_layers[0], embedding_size * [1], 2)  # a list batch * filed_size * 1
    with tf.variable_scope("cin") as scope:
        for idx, layer_size in enumerate(num_layers): # cross_layer_size is set to be 4
            split_tensor = tf.split(hidden_nn_layers[-1], embedding_size * [1], 2) 
            dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)  # embedding_size * bacth_size * filed_size0 * filed_sizek
            dot_result_o = tf.reshape(dot_result_m, shape=[embedding_size, -1, field_sizes[0]*field_sizes[-1]])
            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])  # bacth * embedding_size * (filed_size0 * filed_sizek)
            
            filters = tf.get_variable(name="f_"+str(idx),
                                 shape=[1, field_sizes[-1]*field_sizes[0], layer_size],
                                 dtype=tf.float32)
            # dot_result = tf.transpose(dot_result, perm=[0, 2, 1])
            curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')

            # BIAS ADD
            if bias:
                b = tf.get_variable(name="f_b" + str(idx),
                                shape=[layer_size],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer())
                curr_out = tf.nn.bias_add(curr_out, b)

            curr_out = deep_layers_activation(curr_out)

            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])
            
            # direct: without split?
            if direct:
                direct_connect = curr_out
                next_hidden = curr_out
                final_len += layer_size
                field_sizes.append(int(layer_size))

            else:
                if idx != len(num_layers) - 1:
                    next_hidden, direct_connect = tf.split(curr_out, 2 * [int(layer_size / 2)], 1)
                    final_len += int(layer_size / 2)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
                    final_len += layer_size
                field_sizes.append(int(layer_size / 2))

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

    result = tf.concat(final_result, axis=1)
    result = tf.reduce_sum(result, -1)  # pooling, batch * final_len
    
    return result, final_len

