import tensorflow as tf
 

def dice(_x,axis=-1,epsilon=0.0000001,name=''):

    alphas = tf.get_variable('alpha'+name,_x.get_shape()[-1],
                             initializer = tf.constant_initializer(0.0),
                             dtype=tf.float32)

    input_shape = list(_x.get_shape()) #[batch_size, flied_size * embedding_size]
    reduction_axes = list(range(len(input_shape)))  #[0,1]

    del reduction_axes[axis] # [0]

    broadcast_shape = [1] * len(input_shape)  #[1,1]
    broadcast_shape[axis] = input_shape[axis] # [1, hidden_unit_size], hidden_unit_size=flied_size * embedding_size

    # case: train mode (uses stats of the current batch)
    mean = tf.reduce_mean(_x, axis=reduction_axes) # [1 * hidden_unit_size]
    brodcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
    std = tf.sqrt(std)
    brodcast_std = tf.reshape(std, broadcast_shape) #[1 * hidden_unit_size]
    x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
    # x_normed = tf.layers.batch_normalization(_x, center=False, scale=False)  # a simple way to use BN to calculate x_p
    x_p = tf.sigmoid(x_normed)

    return alphas * (1.0 - x_p) * _x + x_p * _x


def parametric_relu(_x):
    alphas = tf.get_variable('alpha', [_x.get_shape()[-1]],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg


def attention(queries, keys, keys_length):
    '''
        queries, i.e, advertising:     [batch_size, embedings_size]
        keys:        [batch_size, length ,embedings_size], notice length id vary with user, so  keys_length used
        keys_length: [batch_size]
    '''

    # trans queries to [batch_size, length ,embedings_size]
    queries_hidden_units = queries.get_shape().as_list()[-1]
    queries = tf.tile(queries,[1,tf.shape(keys)[1]])
    queries = tf.reshape(queries,[-1,tf.shape(keys)[1],queries_hidden_units])



    din_all = tf.concat([queries,keys,queries-keys,queries * keys],axis=-1) # 
    # 三层全链接
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att')
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att')
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att') #[batch_size, length ,  1]

    outputs = tf.reshape(d_layer_3_all,[-1,1,tf.shape(keys)[1]]) #[batch_size, 1 , length]

    # Mask
    key_masks = tf.sequence_mask(keys_length,tf.shape(keys)[1])  # [batch_size, length]
    key_masks = tf.expand_dims(key_masks,1) # [batch_size, 1 , length]
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1) # a very small nunber, rather than 0
    outputs = tf.where(key_masks,outputs,paddings) # #[batch_size, 1 , length]


    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)


    # Activation
    outputs = tf.nn.softmax(outputs) # [batch_size, 1 , length]

    # Weighted Sum
    outputs = tf.matmul(outputs,keys) # [batch_size, 1 , length]
    
    outputs = tf.reshape(outputs, [-1, queries_hidden_units])

    return outputs

