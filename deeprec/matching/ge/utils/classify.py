# -*- coding: UTF-8 -*-
# 做分类任务,重构的API，在mian函数最后位置被调用
from __future__ import print_function
import numpy
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from time import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pdb


def getSimilarity(result):
    """
    # 得到类似与邻接矩阵一样的矩阵，值为连接的可能性,看排序的准确性
    :param result: embedding, ns * es
    :return:
    """
    print("getting similarity...")
    return np.dot(result, result.T)


def getAdj(g):
    """
    :param g:
    :return:
    """
    node_size = g.node_size
    look_up = g.look_up_dict  # look_up是一个标号dict (node_str, id)，防止就是节点不连续
    adj = np.zeros((node_size, node_size))
    for edge in g.G.edges():
        adj[look_up[edge[0]]][look_up[edge[1]]] = 1
    return adj


def check_reconstruction(embedding, g, check_index):
    """
    :param embedding: 生成的embedding
    :param g:
    :param check_index: 数目，[100，300，500，700，900]
    :return:
    """
    def get_precisionK(embedding, adj_matrix, max_index):
        print("get precisionK...")
        similarity = getSimilarity(embedding).reshape(-1)  # flaten
        sortedInd = np.argsort(similarity)  # 序号

        cur = 0
        count = 0
        precisionK = []
        sortedInd = sortedInd[::-1]  # 从大到小
        #pdb.set_trace()
        for ind in sortedInd:
            x = int(ind / embedding.shape[0])
            y = int(ind % embedding.shape[0])
            count += 1
            if (adj_matrix[x][y] == 1 or x == y):
                cur += 1
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                break
        return precisionK

    adj_matrix = getAdj(g)
    precisionK = get_precisionK(embedding, adj_matrix,  np.max(check_index))
    ret = []
    for index in check_index:
        print("precisonK[%d] %.2f" % (index, precisionK[index - 1]))
        ret.append(precisionK[index - 1])
    return ret



def check_link_prediction(embedding, train_graph, origin_graph, check_index):
    """
    :param embedding: 生成的embedding
    :param train_graph: 去除部分链路的图
    :param origin_graph:  原图
    :param check_index: check_index： 数目，[100，300，500，700，900]
    :return:
    """
    def get_precisionK(embedding, train_adj_matrix, origin_adj_matrix, max_index):
        print( "get precisionK...")
        similarity = getSimilarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)
        cur = 0
        count = 0
        precisionK = []
        sortedInd = sortedInd[::-1]
        for ind in sortedInd:
            x = int(ind / embedding.shape[0])
            y = int(ind % embedding.shape[0])
            if (x == y or train_adj_matrix[x][y] == 1):
                continue
            count += 1
            #  在原图不在训练图中
            if (origin_adj_matrix[x][y] == 1):
                cur += 1
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                break
        return precisionK
    train_adj_matrix = getAdj(train_graph)
    origin_adj_matrix = getAdj(origin_graph)
    #pdb.set_trace()
    precisionK = get_precisionK(embedding, train_adj_matrix, origin_adj_matrix, np.max(check_index))
    ret = []
    for index in check_index:
        print( "precisonK[%d] %.2f" % (index, precisionK[index - 1]))
        ret.append(precisionK[index - 1])
    return ret



class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)


class Classifier(object):

    def __init__(self, vectors, clf):
        self.embeddings = vectors
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        # print('Results, using embeddings of dimensionality', len(self.embeddings[X[0]]))
        # print('-------------------')
        print(results)
        return results
        # print('-------------------')

    def predict(self, X, top_k_list):
        X_ = numpy.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        state = numpy.random.get_state()

        training_size = int(train_precent * len(X))
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        numpy.random.set_state(state)
        return self.evaluate(X_test, Y_test)

    ## 5折交叉验证
    def split_train_evaluate_cross_val(self, X, Y, seed=0):
        state = numpy.random.get_state()
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        length =  len(X)
        x_block_0 = [X[shuffle_indices[i]] for i in range(int(length/5))]
        x_block_1 = [X[shuffle_indices[i]] for i in range(int(length / 5), int(2*length / 5))]
        x_block_2 = [X[shuffle_indices[i]] for i in range(int(2*length / 5), int(3*length / 5))]
        x_block_3 = [X[shuffle_indices[i]] for i in range(int(3*length / 5), int(4*length / 5))]
        x_block_4 = [X[shuffle_indices[i]] for i in range(int(4*length / 5), len(X))]
        X_block = [x_block_0, x_block_1, x_block_2, x_block_3, x_block_4]

        y_block_0 = [Y[shuffle_indices[i]] for i in range(int(length/5))]
        y_block_1 = [Y[shuffle_indices[i]] for i in range(int(length / 5), int(2*length / 5))]
        y_block_2 = [Y[shuffle_indices[i]] for i in range(int(2*length / 5), int(3*length / 5))]
        y_block_3 = [Y[shuffle_indices[i]] for i in range(int(3*length / 5), int(4*length / 5))]
        y_block_4 = [Y[shuffle_indices[i]] for i in range(int(4*length / 5), len(X))]
        Y_block = [y_block_0, y_block_1, y_block_2, y_block_3, y_block_4]

        all_results = {}
        all_results["micro"] = 0
        all_results["macro"] = 0
        all_results["samples"] = 0
        all_results["weighted"] = 0

        for i in range(5):
            x_temp = [var for var in X_block]
            X_test = x_temp[i]
            del x_temp[i]
            X_train = []
            for var in x_temp:
                X_train+= var

            y_temp = [var for var in Y_block]
            Y_test = y_temp[i]
            del y_temp[i]
            Y_train = []
            for var in y_temp:
                Y_train+= var
            self.train(X_train, Y_train, Y)
            temp = self.evaluate(X_test, Y_test)
            all_results["micro"] += temp["micro"]
            all_results["macro"] += temp["macro"]
            all_results["samples"] += temp["samples"]
            all_results["weighted"] += temp["weighted"]

        all_results["micro"] /= 5
        all_results["macro"] /= 5
        all_results["samples"] /= 5
        all_results["weighted"] /= 5
        numpy.random.set_state(state)
        print(all_results)
        return all_results


def load_embeddings(filename):
    fin = open(filename, 'r')
    node_num, size = [int(x) for x in fin.readline().strip().split()]
    vectors = {}
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        assert len(vec) == size+1
        vectors[vec[0]] = [float(x) for x in vec[1:]]
    fin.close()
    assert len(vectors) == node_num
    return vectors


def read_node_label(filename):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y