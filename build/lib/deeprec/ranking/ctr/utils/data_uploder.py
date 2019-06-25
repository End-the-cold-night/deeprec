# -*- coding:utf-8 -*-
"""
Author:
    Kai Zhang, kaizhangee@gmail.com

模块完成从数据中读取一段，返回相应的数据
"""


import numpy as np
import tensorflow as tf



class data_uploder:
    def __init__(self, data, feature_config_dict, batch_size, sess_max_count = 5, sess_len_max =10, use_mask = False):
        self.batch_size = batch_size  # bs
        self.data = data  # include train_set or test_set
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0
        self.sess_max_count = sess_max_count
        self.sess_len_max = sess_len_max
        self.use_mask = use_mask
        self.feature_config_dict = feature_config_dict
        self.cal_feature_number()

    def cal_feature_number(self):
        # 统计sprase, sequence, dense的长度
        self.number_of_sprase_feature = 0
        self.number_of_sequence_feature = 0
        self.number_of_dense_feature = 0
        for var in self.feature_config_dict.keys():
            if var.split("_")[-1] == "sprase":
                self.number_of_sprase_feature += 1
            elif var.split("_")[-1] == "sequence":
                self.number_of_sequence_feature += 1
            elif var.split("_")[-1] == "dense":
                self.number_of_dense_feature += 1
            else:
                print("feature_config_dict is named error")
                raise NameError

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration

        # batch_data, list
        batch_data = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size, len(self.data))]
        self.i += 1

        """
        数据加载
        sprase_data， list, bs个元素，元素内部是tuple, 
        sequence_data, list, bs个元素，元素内部是tuple, tuple的元素是list，
        dense_data， list, bs个元素，元素内部是tuple, 
        target_data， list, bs个元素，1个值
        label_data， list, bs个元素，1个值

        e.g.
            a = [(103944, 103944, [17704, 37473, 33], [17704, 37473], 53346, 0), 
            (126219, 126219, [15082, 19768, 30450], [15082, 30450],48620, 0)]

            t =[]

            for var in a:
                t.append(var[2:4])

            [([17704, 37473, 33], [17704, 37473], []),
             ([15082, 30450], [15082, 30450], [15082, 19768, 30450])]
        """
        sprase_data, sequence_data, dense_data, target_data, label_data = [], [], [], [], []

        for t in batch_data:
            sprase_data.append(t[0:self.number_of_sprase_feature])
            sequence_data.append(
                t[self.number_of_sprase_feature: self.number_of_sprase_feature + self.number_of_sequence_feature])
            dense_data.append(t[
                              self.number_of_sprase_feature + self.number_of_sequence_feature: self.number_of_sprase_feature + self.number_of_sequence_feature + self.number_of_dense_feature])
            label_data.append(t[-1])

        if self.use_mask:
            # 构造sequence_data 矩阵, 根据同bs下最大值补充0, sequence_data， bs * K, (K, T会变)
            user_sess_length = []  # 一维list，记录session维度的mask, 例子返回[2, 3]
            user_behavior_length = []  # 二维度list, 记录[[3, 2, 0],[2, 2, 3]], K * T
            # max_sl = np.zeros([self.number_of_sequence_feature, ])
            for x in sequence_data:
                temp_num = 0
                temp_user_behavior_length = []
                for y in x:
                    if len(y):
                        temp_num += 1
                    temp_user_behavior_length.append(len(y))
                user_sess_length.append(temp_num)
                # 防止输入端忘记补充[]
                while len(temp_user_behavior_length) < self.sess_max_count:
                    temp_user_behavior_length.append(0)
                user_behavior_length.append(temp_user_behavior_length)

            masked_sequence_data = {}
            temp_num = 0
            for var in self.feature_config_dict.keys():
                if var.split("_")[-1] == "sequence":
                    masked_sequence_data[var] = np.zeros([len(batch_data), int(self.sess_len_max)], np.int64)
                    k = 0
                    for x in sequence_data:
                        for y in range(min(len(x[temp_num]), int(self.sess_len_max))):
                            masked_sequence_data[var][k][y] = x[temp_num][y]
                        k += 1
                    temp_num += 1
            return  self.i, (np.array(sprase_data).reshape([len(batch_data), -1]), masked_sequence_data,
                     np.array(dense_data).reshape([len(batch_data), -1]), label_data), (user_sess_length, user_behavior_length)
        else:
            max_sl = np.zeros([self.number_of_sequence_feature, ])
            for x in sequence_data:
                for i in range(len(x)):
                    max_sl[i] = max(max_sl[i], len(x[i]))

            masked_sequence_data = {}
            temp_num = 0
            for var in self.feature_config_dict.keys():
                if var.split("_")[-1] == "sequence":
                    masked_sequence_data[var] = np.zeros([len(batch_data), int(max_sl[temp_num])], np.int64)
                    k = 0
                    for x in sequence_data:
                        for y in range(len(x[temp_num])):
                            masked_sequence_data[var][k][y] = x[temp_num][y]
                        k += 1
                    temp_num += 1
            return self.i, (np.array(sprase_data).reshape([len(batch_data), -1]), masked_sequence_data,
                        np.array(dense_data).reshape([len(batch_data), -1]), label_data)
