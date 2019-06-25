# -*- coding:utf-8 -*-
"""
Author:
    Kai Zhang, kaizhangee@gmail.com

    测试数据加载模块
"""


import numpy as np
import pandas as pd
import pickle

train_set = [(103944, 103944, [17704, 37473], [17704, 37473], 4.5, 53346, 0),
             (126219, 126219, [15082, 19768, 30450], [30450],8.4, 48620, 1)]

test_set = [(103944, 103944, [17704, 37473], [17704, 37473], 4.5, 53346, 0),
             (126219, 126219, [15082, 19768, 30450], [30450],8.4, 48620, 1)]

feature_config_dict = {}
feature_config_dict['uid1_sprase'] = 4
feature_config_dict['uid2_sprase'] = 4
feature_config_dict['sid1_sequence'] = 3
feature_config_dict['sid2_sequence'] = 3
feature_config_dict['did1_dense'] = 4
feature_config_dict['uid2_taregt'] = 4
feature_config_dict['label'] = 0

with open("../data/test_data.pkl", "wb") as f:
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(feature_config_dict, f, pickle.HIGHEST_PROTOCOL)


