"""
Base function 
"""
import tensorflow as tf

def cal_feature_number(feature_config_dict):
    """
    The function is used to count the number of features of various types.
    :param feature_config_dict: feature dict
    :return:
    """
    sequence_feature_name = ""
    number_of_sprase_feature = 0
    number_of_sequence_feature = 0
    number_of_dense_feature = 0
    for var in feature_config_dict.keys():
        if var.split("_")[-1] == "sprase":
            number_of_sprase_feature += 1
        elif var.split("_")[-1] == "sequence":
            number_of_sequence_feature += 1
            sequence_feature_name = var
        elif var.split("_")[-1] == "dense":
            number_of_dense_feature += 1
        else:
            print("feature_config_dict is named error")
            raise NameError
    return number_of_sprase_feature, number_of_sequence_feature, \
           number_of_dense_feature, sequence_feature_name



def get_linear_embedding(feature_config_dict, sprase_data, number_of_sprase_feature):
    """
    This function is used to extract first-order information.
    :param feature_config_dict: feature dict
    :param sprase_data: a tensot, bs * number_of_sprase_feature
    :param number_of_sprase_feature:
    :return: sprase_feature list, the embedding of sprase data
    """
    sprase_feature = []
    for var in feature_config_dict.keys():
        if var.split("_")[-1] == "sprase":
            sprase_feature.append(var)

    embedding_dict = {}
    for var in sprase_feature:
        embedding_var = var[:-7] + "_linear"
        embedding_dict[embedding_var] = tf.get_variable("embedding_{}".format(embedding_var),
                                                        [feature_config_dict[var], 1])

    sprase_data_list = tf.split(sprase_data, number_of_sprase_feature, axis=1)
    sprase_data_embedding_list = []
    temp = 0
    for var in sprase_feature:
        # list, bs * es
        embedding_var = var[:-7] + "_linear"
        sprase_data_embedding_list.append(
            tf.nn.embedding_lookup(embedding_dict[embedding_var], sprase_data_list[temp]))
        temp += 1

    sprase_data_linear_embedding = tf.concat(sprase_data_embedding_list, axis=1)
    sprase_data_linear_embedding = tf.reshape(sprase_data_linear_embedding,
                                                   [-1, number_of_sprase_feature])
    return sprase_feature, sprase_data_linear_embedding



def get_embedding(sprase_feature, feature_config_dict,embedding_size, sprase_data):
    """
    This function is used to extract embedding
    :param sprase_feature: a list, indicate the sprase feature
    :param feature_config_dict: feature dict
    :param embedding_size: i.e., es
    :param sprase_data: a tensot, bs * number_of_sprase_feature
    :return: the embedding of sprase data, bs * fs * es
    """
    number_of_sprase_feature = len(sprase_feature)
    embedding_dict = {}
    for var in sprase_feature:
        embedding_var = var[:-7]
        embedding_dict[embedding_var] = tf.get_variable("embedding_{}".format(embedding_var),
                                                        [feature_config_dict[var], embedding_size])

    sprase_data_list = tf.split(sprase_data, number_of_sprase_feature, axis=1)
    sprase_data_embedding_list = []
    temp = 0
    for var in sprase_feature:
        embedding_var = var[:-7]
        sprase_data_embedding_list.append(
            tf.nn.embedding_lookup(embedding_dict[embedding_var], sprase_data_list[temp]))
        temp += 1
    sprase_data_embedding = tf.concat(sprase_data_embedding_list, axis=1)
    return embedding_dict, sprase_data_embedding



def get_sequence_embedding(embedding_dict, masked_sequence_data, sequence_feature_name, embedding_size):
    """
    :param embedding_dict: embedding_dict from get_embedding
    :param masked_sequence_data: the masked sequence data
    :param sequence_feature_name: sequence feature name
    :param embedding_size: embedding_size,
    :return: a tensor, bs * T * es, T is the max time
    """
    sequence_data_embedding = tf.nn.embedding_lookup(embedding_dict[sequence_feature_name[:-9]],
                                                          masked_sequence_data)
    paddings = tf.zeros_like(sequence_data_embedding)
    mask = tf.expand_dims(masked_sequence_data, axis=-1)
    mask = tf.tile(mask, [1, 1, embedding_size])
    mask = tf.cast(mask, tf.bool)
    sequence_data_embedding = tf.where(mask, sequence_data_embedding, paddings)
    return sequence_data_embedding