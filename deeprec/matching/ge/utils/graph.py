# -*- coding: UTF-8 -*-
"""
Ref: https://github.com/thunlp/OpenNE
在networkx基础上做了一层封装
"""
from time import time
import networkx as nx
import pickle as pkl
import numpy as np
import scipy.sparse as sp



class Graph(object):
    def __init__(self):
        self.G = None   # 一个和nx的g一样的object
        self.look_up_dict = {}  # 一个节点和数字的映射，就是编号
        self.look_back_list = []  # 节点顺序的一个集合
        self.node_size = 0

    def encode_node(self):
        """
        对节点进行编码，look_up_dict是节点的编码，利用list顺序特性look_back_list是反编码
        :return:
        """
        look_up = self.look_up_dict
        look_back = self.look_back_list
        for node in self.G.nodes():
            look_up[node] = self.node_size
            look_back.append(node)
            self.node_size += 1
            self.G.nodes[node]['status'] = ''

    def read_g(self, g):
        self.G = g
        self.encode_node()


    def read_adjlist(self, filename):
        """
        直接读取adj矩阵，无需判读是否有方向，图必须是没有权重的
        the format of each line: v1 n1 n2 n3 ... nk
        :param filename: 类似data/wiki/wiki_edgelist.txt
        :return: None
        """
        self.G = nx.read_adjlist(filename, create_using=nx.DiGraph())
        for i, j in self.G.edges():
            self.G[i][j]['weight'] = 1.0
        self.encode_node()

    def read_edgelist(self, filename, weighted=False, directed=False):
        """
        从三元祖读取图, 要求无权重的输入为(node1, node2), 有权重的输入为(node1, node2, weight)
        :param filename: 类似data/wiki/wiki_edgelist.txt
        :param weighted: 权重
        :param directed: 方向
        :return:
        """
        self.G = nx.DiGraph()

        if directed:
            def read_unweighted(l):
                src, dst = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = 1.0

            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = float(w)
        else:
            def read_unweighted(l):
                src, dst = l.split()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = 1.0
                self.G[dst][src]['weight'] = 1.0

            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = float(w)
                self.G[dst][src]['weight'] = float(w)
        fin = open(filename, 'r')
        func = read_unweighted
        if weighted:
            func = read_weighted
        while 1:
            l = fin.readline()
            if l == '':
                break
            func(l)
        fin.close()
        self.encode_node()

    def read_node_label(self, filename):
        """
        读取标签，文件要求第一个是节点，后面是节点的标签编码，node, 0,0,1,1,0
        :param filename: 文件位置
        :return:
        """
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[vec[0]]['label'] = vec[1:]
        fin.close()


    def read_node_features(self, filename):
        """
        有单独的feature文件, 文件要求第一个是节点，后面是节点的特性编码，node, 0.4, 0.3, 4.5...
        :param filename:
        :return:
        """
        fin = open(filename, 'r')
        for l in fin.readlines():
            vec = l.split()
            self.G.nodes[vec[0]]['feature'] = np.array(
                [float(x) for x in vec[1:]])
        fin.close()

    def read_node_status(self, filename):
        """
        节点的状态标签，格式 node, status
        :param filename:
        :return:
        """
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[vec[0]]['status'] = vec[1]  # train test valid
        fin.close()

    def read_edge_label(self, filename):
        """
        边的标签，格式文件要求第一个是边的标号，后面是节点的标签编码，node, 0,0,1,1,0
        :param filename:
        :return:
        """
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G[vec[0]][vec[1]]['label'] = vec[2:]
        fin.close()