# -*- coding: UTF-8 -*-
"""
Author:
    Kai Zhang, kaizhangee@gmail.com
"""
from __future__ import print_function
from gensim.models import Word2Vec
from ..utils.walker import BasicWalker, Walker
import numpy as np

class DeepWalk(object):

    def __init__(self, graph, path_length, num_paths, **kwargs):
        """
        :param graph: a networkx object
        :param path_length: 
        :param num_paths: 
        :param embedding_dim: 
        :param kwargs: 
        """
        kwargs["workers"] = kwargs.get("workers", 1)
        self.graph = graph.G
        self.walker = BasicWalker(graph, workers=kwargs["workers"])
        self.sentences = self.walker.simulate_walks(
            num_walks=num_paths, walk_length=path_length)

    def train(self, embedding_size=128, window_size=5, workers=3, iter=5, filename= "", save_embedding = True, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = kwargs.get("size", embedding_size)
        kwargs["sg"] = 1
        kwargs["hs"] = 0
        kwargs["window"] = window_size
        kwargs["iter"] = iter

        self.size = kwargs["size"]
        print("Learning representation...")
        self.model = Word2Vec(**kwargs)

        self.vectors = {}
        for word in self.graph.nodes():
            self.vectors[word] = self.model.wv[word]

        self.embedding =[]

        if save_embedding:
            if filename=="":
                filename = "../result/"  + str(self.__class__)
            fout = open(filename, 'w+')
            node_num = len(self.vectors.keys())
            fout.write("{} {}\n".format(node_num, self.size))
            for node, vec in self.vectors.items():
                self.embedding.append(np.array(vec))
                fout.write("{} {}\n".format(node,
                                            ' '.join([str(x) for x in vec])))
            fout.close()

        return self.model, np.array(self.embedding), self.vectors