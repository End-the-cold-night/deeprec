# -*- coding:utf-8 -*-
"""
Author:
    Kai Zhang, kaizhangee@gmail.com
"""
import tensorflow as tf
import numpy as np
from .sequence import *


def fm(embeddings):
    """
    :param embeddings: embeddings should be,  bs * fs * es,
    :return: second order embedding,  bs * es
    """
    summed_features_emb = tf.reduce_sum(embeddings, 1)  # bs * es
    summed_features_emb_square = tf.square(summed_features_emb)  # bs * es
    squared_features_emb = tf.square(embeddings)
    squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # bs * es
    y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # bs * es
    return y_second_order



def dnn(input, dnn_hidden_units, dnn_activation=tf.nn.sigmoid, regularizer_rate=0.0,
        use_dropout=0.5, use_bn=True, seed=1024):
    """
    :param input: input, 输入，格式 bs * input_size
    :param dnn_hidden_units: dnn_hidden_units, list, 代表每层的hidden unit number
    :param dnn_activation:  dnn_activation
    :param regularizer_rate:
    :param use_dropout:
    :param use_bn:
    :param seed:
    :return:
    """
    if(len(input.get_shape().as_list()))==3:
        input = tf.reshape(input, [-1, input.get_shape().as_list()[1] * input.get_shape().as_list()[2]])
    for i in range(len(dnn_hidden_units)):
        input = tf.layers.dense(input, dnn_hidden_units[i], activation=dnn_activation,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate))
        if use_bn:
            input = tf.layers.batch_normalization(inputs=input)
        if use_dropout!=0:
            input = tf.nn.dropout(input, use_dropout)

        return input



def afm(embeddings, attention_size):
    """
    :param embeddings: embeddings: bs * fs * es
    :param attention_size: attention_size,
    :return:
    """
    field_size = embeddings.get_shape().as_list()[1]
    embedding_size = embeddings.get_shape().as_list()[2]

    weights = dict()
    # attention part
    glorot = np.sqrt(2.0 / (attention_size + embedding_size))
    weights['attention_w'] = tf.Variable(np.random.normal(loc=0, scale=glorot,
                                                          size=(embedding_size, attention_size)),
                                         dtype=tf.float32, name='attention_w')
    weights['attention_b'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(attention_size,)),
                                         dtype=tf.float32, name='attention_b')
    weights['attention_h'] = tf.Variable(np.random.normal(loc=0, scale=1, size=(attention_size,)),
                                         dtype=tf.float32, name='attention_h')
    weights['attention_p'] = tf.Variable(np.ones((embedding_size, 1)), dtype=np.float32)

    # element_wise
    element_wise_product_list = []
    for i in range(field_size):
        for j in range(i + 1, field_size):
            element_wise_product_list.append(tf.multiply(embeddings[:, i, :], embeddings[:, j, :]))  # bs * es
    element_wise_product = tf.stack(element_wise_product_list)  # (fs * fs * 1 / 2) * bs * es
    element_wise_product = tf.transpose(element_wise_product, perm=[1, 0, 2],name='element_wise_product')  # bs * (fs * fs * 1 / 2) *  es

    # attention part
    num_interactions = int(field_size * (field_size - 1) / 2)
    attention_wx_plus_b = tf.reshape(tf.add(tf.matmul(
        tf.reshape(element_wise_product,shape=(-1, embedding_size)),weights['attention_w']),
        weights['attention_b']),shape=[-1, num_interactions, attention_size])
    # bs * ( fs * fs * 1 / 2) * as
    attention_exp = tf.exp(tf.reduce_sum(tf.multiply(tf.nn.relu(attention_wx_plus_b),
                                                     weights['attention_h']),
                                         axis=2, keep_dims=True))
    #  bs * ( fs * fs * 1 / 2) * 1
    attention_exp_sum = tf.reduce_sum(attention_exp, axis=1, keep_dims=True)  # bs * 1 * 1
    attention_out = tf.div(attention_exp, attention_exp_sum, name='attention_out')  # bs * ( fs * fs * 1 / 2) * 1
    attention_x_product = tf.reduce_sum(tf.multiply(attention_out, element_wise_product), axis=1, name='afm')  # bs * es
    return attention_x_product



def cin(inputs, layer_size=(128, 128), activation_funtion=tf.nn.relu, split_half=True, l2_reg=1e-5, seed=1024):
    """
    :param inputs: input, bs * fs * es
    :param layer_size: cin layer
    :param activation:
    :param split_half:
    :param l2_reg:
    :param seed:
    :return:
    """

    field_nums = [inputs.get_shape().as_list()[1]]
    embedding_size = inputs.get_shape().as_list()[2]


    """
    wight defination
    """
    filters = []
    bias = []
    for i, size in enumerate(layer_size):
        glorot = np.sqrt(2.0 / (field_nums[-1] * field_nums[0] + size))

        filters.append(tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, field_nums[-1] * field_nums[0], size)),
                dtype=tf.float32))

        bias.append(tf.Variable(np.random.normal(loc=0, scale=glorot, size=(size)),dtype=tf.float32))

        if split_half:
            if size%2>0:
                raise ValueError("cin layer size must be even number")
            field_nums.append(size // 2)
        else:
            field_nums.append(size)


    hidden_nn_layers = [inputs]  # inputs: bs * fs * es
    final_result = []

    split_tensor0 = tf.split(hidden_nn_layers[0], embedding_size * [1], 2)  # list , es个，每个元素是 bs * fs0 * 1
    for idx, size in enumerate(layer_size):
        split_tensor = tf.split(hidden_nn_layers[-1], embedding_size * [1], 2)
        # list , 每个元素是 bs * fsn * 1

        dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
        # 会自动转为一个完整的， es * bs * fs0 * fsn

        dot_result_o = tf.reshape(
            dot_result_m, shape=[embedding_size, -1, field_nums[0] * field_nums[idx]])  # es * bs * (fs0 * fsn

        dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])  # bs * es * (fs0 * fsn)


        curr_out = tf.nn.conv1d(dot_result, filters=filters[idx], stride=1, padding='VALID')  # bs * es * fs(n+1)

        curr_out = tf.nn.bias_add(curr_out, bias[idx])


        curr_out = activation_funtion(curr_out)


        curr_out = tf.transpose(curr_out, perm=[0, 2, 1])  # bs * fs(n+1) * es

        if split_half:
            if idx != len(layer_size) - 1:
                next_hidden, direct_connect = tf.split(
                    curr_out, 2 * [size // 2], 1)
            else:
                direct_connect = curr_out
                next_hidden = 0
        else:
            direct_connect = curr_out
            next_hidden = curr_out

        final_result.append(direct_connect)
        hidden_nn_layers.append(next_hidden)

    result = tf.concat(final_result, axis=1)
    result = tf.reduce_sum(result, -1, keep_dims=False)

    return result



def ipnn(embeddings, reduce_sum):
    """
    :param embeddings: bs * fs * es
    :param reduce_sum: whether to sum up on es
    :return: if reduce_sum, bs * (fs * (fs-1)/ 2), else bs * (fs * (fs-1)/ 2) * es
    """

    # To split, every tensor is bs * 1 * es
    filed_size = embeddings.get_shape().as_list()[1]
    embed_list = tf.split(embeddings, num_or_size_splits = filed_size, axis = 1)
    row = []
    col = []

    for i in range(filed_size - 1):
        for j in range(i + 1, filed_size):
            row.append(i)
            col.append(j)

    # bs * (fs * (fs-1)/ 2) * es
    p = tf.concat([embed_list[idx] for idx in row], axis=1)

    # bs * (fs * (fs-1)/ 2) * es
    q = tf.concat([embed_list[idx] for idx in col], axis=1)

    inner_product = p * q  # bs * (fs * (fs-1)/ 2) * es

    if reduce_sum:
        inner_product = tf.reduce_sum(inner_product, axis=2)

    return inner_product



def opnn(embeddings, deep_init_size):
    """
    :param embeddings: bs * fs * es
    :param deep_init_size: number of deep size
    :return: if reduce_sum, bs * (fs * (fs-1)/ 2), else bs * (fs * (fs-1)/ 2) * es
    """
    embedding_size = embeddings.get_shape().as_list()[-1]
    embedding_sum = tf.reduce_sum(embeddings, axis=1)
    p = tf.matmul(tf.expand_dims(embedding_sum,2),tf.expand_dims(embedding_sum,1)) # bs * es * es
    p = tf.reshape(p, [-1, embedding_size * embedding_size])
    ret = []
    for i in range(deep_init_size):
        t = tf.layers.dense(p, embedding_size * embedding_size)
        t = tf.reduce_sum(t, axis=1)
        t = tf.reshape(t, [-1, 1])
        ret.append(t)

    return tf.concat(ret, axis = 1)  # bs * deep_init_size



def cross(embeddings, layer_num=2):
    """
    :param embeddings: input, bs * fs * es or  bs * (fs * es)
    :param layer_num: 
    :return: bs * (fs * es)
    """
    if len(embeddings.get_shape().as_list())==3:
        x_0 = tf.reshape(embeddings,
                         [-1, embeddings.get_shape().as_list()[1] * embeddings.get_shape().as_list()[2], 1])
    else:
        x_0 = tf.expand_dims(embeddings, axis=2)  # bs * (fs * es) * 1

    # kernels, bias, defination
    kernels, bias = {}, {}
    glorot = np.sqrt(1.0 / (x_0.get_shape().as_list()[1]))
    for i in range(layer_num):
        kernels[i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(x_0.get_shape().as_list()[1], 1)),
                                         dtype=tf.float32)
        bias [i] =tf.Variable(np.random.normal(loc=0, scale=glorot, size=(x_0.get_shape().as_list()[1], 1)),
                                         dtype=tf.float32)

    x_l = x_0
    for i in range(layer_num):
        xl_w = tf.tensordot(x_l, kernels[i], axes=(1, 0))  # bs * 1 * 1
        dot_ = tf.matmul(x_0, xl_w)  # bs * (fs * es) * 1
        x_l = dot_ + bias[i] + x_l
    x_l = tf.squeeze(x_l, axis=2)

    return x_l



def ffm(inputs, filed_size, embedding_size, feature_config_dict):
    """
    :param inputs:  注意和embedding不同，这里的输入是0，1的
    :param filed_size:
    :param embedding_size:
    :param feature_config_dict:
    :return: bs * 1
    """
    feature_size = inputs.get_shape().as_list()[1]

    feature_size_2_filed_size = {}
    filed_size_list = []
    for var in feature_config_dict:
        if var.split("_")[-1] == "sprase":
            filed_size_list.append(feature_config_dict[var])
    key = 0
    filed_size_list = np.cumsum(filed_size_list)

    if feature_size != filed_size_list[-1]:
        raise ValueError

    for i in range(filed_size_list[-1]):
        if i == filed_size_list[key]:
            key += 1
        feature_size_2_filed_size[i] = key

    v = tf.get_variable('v', shape=[feature_size, filed_size, embedding_size], dtype='float32',
                        initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
    # shape of [None, 1]
    filed_sizeield_aware_interaction_terms = tf.constant(0, dtype='float32')

    # build dict to find f, key of feature, value of field
    for i in range(feature_size):
        for j in range(i + 1, feature_size):
            filed_sizeield_aware_interaction_terms += tf.multiply(
                tf.reduce_sum(tf.multiply(v[i, feature_size_2_filed_size[i]], v[j, feature_size_2_filed_size[j]])),
                tf.multiply(inputs[:, i], inputs[:, j])
            )

    return tf.reshape(filed_sizeield_aware_interaction_terms, [-1, 1])  # bs * 1




def nffm(sprase_data_list, embedding_size, feature_config_dict):
    """
    :param sprase_data_list:  list, bs * 1
    :param embedding_size:
    :param feature_config_dict:
    :return: bs * (fs * (fs -1) /2) * es
    """

    sprase_feature = []
    for var in feature_config_dict.keys():
        if var.split("_")[-1] == "sprase":
            sprase_feature.append(var)

    if len(sprase_data_list)!=len(sprase_feature):
        raise ValueError

    # embedding
    embedding_dict = {}
    for i in range(len(sprase_feature)):
        embedding_dict[i] = {}
        for j in range(len(sprase_feature)):
            v = tf.get_variable("embedding_{}_{}".format(i, j),
                                [feature_config_dict[sprase_feature[i]], embedding_size])
            embedding_dict[i][j] = v

    nffm_list = []
    for i in range(len(sprase_data_list)):
        for j in range(i+1, len(sprase_data_list)):
            element_wise_prod = tf.multiply(
                tf.nn.embedding_lookup(embedding_dict[i][j], sprase_data_list[i]),
                 tf.nn.embedding_lookup(embedding_dict[j][i], sprase_data_list[j]))
            nffm_list.append(element_wise_prod) # bs * 1 * es

    nffm_out = tf.concat(nffm_list, 1)  # bs * (fs * (fs -1) /2) * es

    return nffm_out



def ccpm(sprase_data_embedding, conv_kernel_width=(2, 1), conv_filters=(4, 4)):
    """
    :param sprase_data_embedding:  bs * fs * es
    :param conv_kernel_width:
    :param conv_filters:
    :return: bs * out_shape
    """
    n, l = sprase_data_embedding.get_shape().as_list()[1], len(conv_kernel_width)
    # bs * fs * es * 1, 构造(bs,  width,  hight,  inchneel)
    sprase_data_embedding = tf.expand_dims(sprase_data_embedding, -1)
    for i in range(1, l+1):
        sprase_data_embedding =  tf.layers.conv2d(sprase_data_embedding, conv_filters[i-1],
                                                  (conv_kernel_width[i-1], 1))

        k = max(1, int((1 - pow(i / l, l - i)) * n)) if i < l else 3

        perm = list(range(len(sprase_data_embedding.get_shape().as_list())))
        # 在fs 维度选择
        perm[-1], perm[1] = perm[1], perm[-1]
        shifted_input = tf.transpose(sprase_data_embedding, perm)

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=min(k, shifted_input.get_shape().as_list()[-1]),
                            sorted=True, name=None)[0]
        sprase_data_embedding = tf.transpose(top_k, perm)

    return tf.layers.flatten(sprase_data_embedding)


def fgcnn(embedding, conv_kernel_width, conv_filters, rerecombination_filters, pooling_width):
    """
    :param embedding:  bs * fs * es
    :param conv_kernel_width:  宽带
    :param conv_filters: 输出channel数目
    :param rerecombination_filters:  重构channel数目
    :param pooling_width: 池化大小
    :return:
    """
    ret = []
    embedding = tf.expand_dims(embedding, -1)
    for i in range(len(conv_kernel_width)):
        # conv layer
        embedding = tf.layers.conv2d(embedding, conv_filters[i], (conv_kernel_width[i], 1))
        # pooling
        embedding = tf.layers.max_pooling2d(embedding, [pooling_width[i],1], 1)
        #rerecombination
        shape_list = embedding.get_shape().as_list()
        embedding_dense = tf.reshape(embedding, [-1, shape_list[1]* shape_list[2] * shape_list[3]])
        embedding_dense = tf.layers.dense(embedding_dense,
                                          shape_list[1]* shape_list[2] *rerecombination_filters[i])
        embedding_dense = tf.reshape(embedding_dense,
                                     [-1, shape_list[1],  shape_list[2], rerecombination_filters[i]])
        embedding_dense = tf.transpose(embedding_dense, [0, 3, 1, 2])
        shape_list = embedding_dense.get_shape().as_list()
        embedding_dense = tf.reshape(embedding_dense, [-1, shape_list[1] * shape_list[2], shape_list[3]])  # bs * new_fs * es
        ret.append(embedding_dense)

    output = tf.concat(ret, axis =1)
    return output





def dice(_x, axis=-1, epsilon=0.0000001, name=''):
    alphas = tf.get_variable('alpha' + name, _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)

    input_shape = list(_x.get_shape())  # [batch_size, flied_size * embedding_size]
    reduction_axes = list(range(len(input_shape)))  # [0,1]

    del reduction_axes[axis]  # [0]

    broadcast_shape = [1] * len(input_shape)  # [1,1]
    broadcast_shape[axis] = input_shape[axis]  # [1, hidden_unit_size], hidden_unit_size=flied_size * embedding_size

    # case: train mode (uses stats of the current batch)
    mean = tf.reduce_mean(_x, axis=reduction_axes)  # [1 * hidden_unit_size]
    brodcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
    std = tf.sqrt(std)
    brodcast_std = tf.reshape(std, broadcast_shape)  # [1 * hidden_unit_size]
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



def din(queries, keys, keys_length=None, Local_use = False, name=""):
    """
    :param queries: bs * es
    :param keys: bs * length * es
    :param Local_use: 只是用于求 lacal attention, 用于dien
    :return:
    """

    # trans queries to [bs, length, es]
    _, length, embedding_size = keys.get_shape().as_list()
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], embedding_size])

    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)  # bs * length * es

    # dnn
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=dice, name='f1_att_{}'.format(name))
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=dice, name='f2_att_{}'.format(name))
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att_{}'.format(name))  # [batch_size, length ,  1]

    outputs = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])  # [bs, 1, length]

    # Mask
    if keys_length!=None:
        key_masks = tf.sequence_mask(keys_length,tf.shape(keys)[1])  # [bs, length]
        key_masks = tf.expand_dims(key_masks,1) # [bs, 1 , length]
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1) # a very small nunber, rather than 0
        outputs = tf.where(key_masks,outputs,paddings)  #[bs, 1 , length]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)  #[bs, 1 , length]


    # Activation
    outputs = tf.nn.softmax(outputs)  #[bs, 1 , length]

    if Local_use:
        return outputs

    # Weighted Sum
    outputs = tf.matmul(outputs, keys)  # [bs, 1 , es]

    outputs_list = outputs.get_shape().as_list()
    queries_hidden_units = 1
    for i in range(1, len(outputs_list)):
        queries_hidden_units *= outputs_list[i]

    outputs = tf.reshape(outputs, [-1, queries_hidden_units])

    return outputs




def transfomer(embeddings, head_num=2, att_embedding_size=8, use_res=True, name=0):
    """
    :param embeddings:  bs * fs * es
    :param head_num:
    :param att_embedding_size:
    :param use_res:  bs * fs * (as * head_num)
    :param name: 防止循环时候变量重名字
    :return:
    """

    embedding_size = embeddings.get_shape().as_list()[-1]

    W_Query = tf.get_variable(name='query_{}'.format(name), shape=[embedding_size, att_embedding_size * head_num],
                                   dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.1, seed=1))
    W_key = tf.get_variable(name='key_{}'.format(name), shape=[embedding_size, att_embedding_size * head_num],
                                   dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.1, seed=1))
    W_Value = tf.get_variable(name='value_{}'.format(name), shape=[embedding_size, att_embedding_size * head_num],
                                   dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.1, seed=1))
    if use_res:
        W_Res = tf.get_variable(name='res_{}'.format(name), shape=[embedding_size, att_embedding_size * head_num],
                                   dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.1, seed=1))

    querys = tf.tensordot(embeddings, W_Query, axes=(-1, 0))  # bs * fs * (as * head_num)
    keys = tf.tensordot(embeddings, W_key, axes=(-1, 0))  # bs * fs * (as * head_num)
    values = tf.tensordot(embeddings, W_Value, axes=(-1, 0))  # bs * fs * (as * head_num)

    querys = tf.stack(tf.split(querys, head_num, axis=2))  # head_num * bs  * fs * as
    keys = tf.stack(tf.split(keys, head_num, axis=2))  # head_num * bs  * fs * as
    values = tf.stack(tf.split(values, head_num, axis=2))  # head_num * bs  * fs * as

    inner_product = tf.matmul(querys, keys, transpose_b=True)  # head_num * bs * fs * fs
    normalized_att_scores = tf.nn.softmax(inner_product)  # head_num * bs * fs * fs

    result = tf.matmul(normalized_att_scores, values)  # head_num * bs * fs * as

    result = tf.concat(tf.split(result, head_num, ), axis=-1)  # 1 * bs * fs * (as * head_num)
    result = tf.squeeze(result, axis=0)  # bs * fs * (as * head_num)

    if use_res:
        result += tf.tensordot(embeddings, W_Res, axes=(-1, 0))  # bs * fs * (as * head_num)
    result = tf.nn.relu(result)

    return result

