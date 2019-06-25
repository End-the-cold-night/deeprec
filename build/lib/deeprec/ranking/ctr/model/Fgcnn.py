# -*- coding:utf-8 -*-
"""
Author:
    Kai Zhang, kaizhangee@gmail.com
"""

import tensorflow as tf
from ..layer.common_layer import fgcnn, ipnn, dnn
from ..utils.utize import cal_feature_number, get_linear_embedding, get_embedding, get_sequence_embedding


class Fgcnn(object):

    def __init__(self, feature_config_dict, embedding_size=3, l2_reg_linear=0.00001,
                  conv_kernel_width=(2, 1), conv_filters=(4, 4),
                 rerecombination_filters=(2, 2), pooling_width=(1, 1), dnn_hidden_units=(20, 10),
                 task='binary'):

        self.feature_config_dict = feature_config_dict
        self.number_of_sprase_feature ,self.number_of_sequence_feature ,\
        self.number_of_dense_feature, self.sequence_feature_name = cal_feature_number(self.feature_config_dict)
        self.sprase_data = tf.placeholder(tf.int32, [None, self.number_of_sprase_feature])
        self.masked_sequence_data = tf.placeholder(tf.int32, [None, None])  # bs * T
        self.dense_data = tf.placeholder(tf.float32, [None, self.number_of_dense_feature])
        self.label = tf.placeholder(tf.float32, [None, ])
        self.lr = tf.placeholder(tf.float64, [])

        # Linear part
        sprase_feature, self.sprase_data_linear_embedding = \
            get_linear_embedding(self.feature_config_dict,self.sprase_data, self.number_of_sprase_feature)

        # sprase embedding
        embedding_dict = {}
        for var in sprase_feature:
            embedding_var = var[:-7]
            embedding_dict[embedding_var] = tf.get_variable("embedding_{}".format(embedding_var),
                                                            [self.feature_config_dict[var], embedding_size])
        embedding_dict_fgcnn = {}
        for var in sprase_feature:
            embedding_var = var[:-7]
            embedding_dict_fgcnn[embedding_var] = tf.get_variable("embedding_fgcnn_{}".format(embedding_var),
                                                            [self.feature_config_dict[var], embedding_size])
        sprase_data_list = tf.split(self.sprase_data, self.number_of_sprase_feature, axis=1)
        sprase_data_embedding_list = []
        sprase_data_embedding_list_fgcnn = []
        temp = 0
        for var in sprase_feature:
            # list, bs * es
            embedding_var = var[:-7]
            sprase_data_embedding_list.append(
                tf.nn.embedding_lookup(embedding_dict[embedding_var], sprase_data_list[temp]))
            sprase_data_embedding_list_fgcnn.append(
                tf.nn.embedding_lookup(embedding_dict[embedding_var], sprase_data_list[temp]))
            temp += 1
        self.sprase_data_embedding = tf.concat(sprase_data_embedding_list, axis=1)  # bs * fs * es
        self.sprase_data_embedding_fgcnn = tf.concat(sprase_data_embedding_list_fgcnn, axis=1)  # bs * fs * es


       # Fgcnn
        fgcnn_out = fgcnn(self.sprase_data_embedding_fgcnn, conv_kernel_width, conv_filters, rerecombination_filters, pooling_width)

        # 合并两种
        out = tf.concat([fgcnn_out, self.sprase_data_embedding], axis = 1)  # bs * new_fs * es

        out  = ipnn(out, 1)  # bs * num



        """
        sequence data
        """
        if self.number_of_sequence_feature:
            # bs * T * es
            self.sequence_data_embedding = tf.nn.embedding_lookup(embedding_dict[self.sequence_feature_name[:-9]],
                                                                  self.masked_sequence_data)
            # mask
            # self.sequence_data_embedding = tf.transpose(self.sequence_data_embedding, (0, 2, 1))
            paddings = tf.zeros_like(self.sequence_data_embedding)
            self.mask = tf.expand_dims(self.masked_sequence_data, axis=-1)
            self.mask = tf.tile(self.mask, [1, 1, embedding_size])  # bs * T * es
            self.mask = tf.cast(self.mask, tf.bool)

            self.sequence_data_embedding = tf.where(self.mask, self.sequence_data_embedding, paddings)
            # FM中对sequence_data_embedding直接在 T 维度做平均了
            self.sequence_data_embedding = tf.reduce_mean(self.sequence_data_embedding, axis=1)  # bs * es
            self.sequence_out = tf.layers.dense(self.sequence_data_embedding, embedding_size)


        out = tf.concat([out, self.sprase_data_linear_embedding], axis =1)

        if self.number_of_sequence_feature:
            out = tf.concat([out, self.sequence_out], axis=1)

        if self.number_of_dense_feature:
            out = tf.concat([out, self.dense_data], axis=1)


        """
        dnn
        """
        out = dnn(out, dnn_hidden_units)

        self.logits = tf.layers.dense(out, 1, activation=None,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_linear))

        self.logits = tf.reshape(self.logits, [-1, ])

        self.pridict = tf.nn.sigmoid(self.logits)

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = \
            tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
            tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

        regulation_rate = 0.0
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.label)
        )

        trainable_params = tf.trainable_variables()
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, uij, l):
        if self.number_of_sequence_feature > 1:
            raise NotImplementedError
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.sprase_data: uij[0],
            self.masked_sequence_data: uij[1][self.sequence_feature_name],
            self.dense_data: uij[2],
            self.label: uij[-1],
            self.lr: l,
        })
        return loss

    def _eval(self, sess, uij):
        if self.number_of_sequence_feature > 1:
            raise NotImplementedError
        pridict = sess.run(self.pridict, feed_dict={
            self.sprase_data: uij[0],
            self.masked_sequence_data: uij[1][self.sequence_feature_name],
            self.dense_data: uij[2],
        })
        return pridict

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)

