import tensorflow as tf
from .fm import fm
import numpy as np

def afm(embeddings, field_size, attention_size, embedding_size):
    weights = dict()
    # attention part
    glorot = np.sqrt(2.0 / (attention_size + embedding_size))
    weights['attention_w'] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(embedding_size,attention_size)),
                                         dtype=tf.float32,name='attention_w')
    weights['attention_b'] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(attention_size,)),
                                         dtype=tf.float32,name='attention_b')
    weights['attention_h'] = tf.Variable(np.random.normal(loc=0,scale=1,size=(attention_size,)),
                                         dtype=tf.float32,name='attention_h')
    weights['attention_p'] = tf.Variable(np.ones((embedding_size,1)),dtype=np.float32)
    
    # element_wise
    element_wise_product_list = []
    for i in range(field_size):
        for j in range(i+1,field_size):
            element_wise_product_list.append(tf.multiply(embeddings[:,i,:],embeddings[:,j,:])) # None * K
    element_wise_product = tf.stack(element_wise_product_list) # (F * F * 1 / 2) * None * K
    element_wise_product = tf.transpose(element_wise_product,perm=[1,0,2],name='element_wise_product') # None * (F * F * 1 / 2) *  K
    #interaction
    # attention part
    num_interactions = int(field_size * (field_size - 1) / 2)
    # wx+b -> relu(wx+b) -> h*relu(wx+b)
    attention_wx_plus_b = tf.reshape(tf.add(tf.matmul(tf.reshape(element_wise_product,shape=(-1,embedding_size)),
                                                           weights['attention_w']),
                                                 weights['attention_b']),
                                          shape=[-1,num_interactions,attention_size]) # N * ( F * F * 1 / 2) * A
    attention_exp = tf.exp(tf.reduce_sum(tf.multiply(tf.nn.relu(attention_wx_plus_b),
                                                   weights['attention_h']),
                                       axis=2,keep_dims=True)) # N * ( F * F * 1 / 2) * 1
    attention_exp_sum = tf.reduce_sum(attention_exp,axis=1,keep_dims=True) # N * 1 * 1
    attention_out = tf.div(attention_exp,attention_exp_sum,name='attention_out')  # N * ( F * F * 1 / 2) * 1
    attention_x_product = tf.reduce_sum(tf.multiply(attention_out,element_wise_product),axis=1,name='afm') # N * K
    return attention_x_product 

