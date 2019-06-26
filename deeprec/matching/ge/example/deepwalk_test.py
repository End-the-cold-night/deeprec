# -*- coding: UTF-8 -*-
"""
Author:
    Kai Zhang, kaizhangee@gmail.com
"""

from __future__ import print_function
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import deeprec
from deeprec.matching.ge.model import DeepWalk
from deeprec.matching.ge.utils import  Graph, Classifier, read_node_label, check_link_prediction, check_reconstruction
import time
import ast
import copy
import pdb





def config():
    args = {}
    # 定义输入格式，adj or triple, adjlist or edgelist
    args["graph_format"] = 'adjlist'
    # 输入文件位置
    args["input"] = "../data/wiki/wiki_edgelist.txt"
    args["label_file"] = "../data/wiki/wiki_labels.txt"
    args["output"] = "../result/"
    args["feature_file"] = ""

    # 表示参数
    args["embedding_size"] = 128
    # 有向图, 权重
    args["directed"] = False
    args["weighted"] = False
    # 游走参数
    args["path_length"] = 80
    args["number_walks"] = 10
    # 训练参数
    args["workers"] = 8
    args["window_size"] = 10
    args["negative_ratio"] = 5
    args["no_auto_save"] = False
    args["model"] = "DeepWalk"

    ## node2vec参数
    args["p"] = 1
    args["q"] = 1

    ## line
    args["order"] = 3

    ## sdne
    args["encoder_list"] = [1000, 128]
    args["lr"] = 0.01
    args["epochs"] = 5
    # sdne2
    args["max_iter"] = 2000
    args["lamb"] = 0.2
    args["beta"] = 5
    args["nu1"] = 1e-5
    args["nu2"] = 1e-4
    args["bs"] = 200

    # grarep
    args["kstep"] = 4

    # hope, i.e., the hope type, include common_neighbors, katz, adamic_adar
    args["hope_type"] = "common_neighbors"
    args["dropout"] = 0.5
    args["decay"] = 5e-4
    args["hidden"] = 16

    # tadw
    args["alpha"] = 1e-6

    args["check_reconstruction"] = True
    args["clf_ratio"] = 0.2
    args["check_link_pridiction"] = False
    args["link_ratio"] = 0.2
    return args


def main(args):
    t1 = time.time()
    g = Graph()
    print("Reading...")

    # 读取数据
    if args["graph_format"] == 'adjlist':
        g.read_adjlist(filename=args["input"])
    elif args["graph_format"] == 'edgelist':
        g.read_edgelist(filename=args["input"], weighted=args["weighted"],directed=args["directed"])


    # 是否做链路测试，做需要将数据集分开
    origin_graph = copy.deepcopy(g)
    if args["check_link_pridiction"]:
        test_ratio = args.link_ratio
        train_set, test_edges = train_test_split(np.array(g.G.edges()), test_size=test_ratio)
        print('#test edges: {}'.format(len(test_edges)))
        # 从图中删除测试边，注意因为读取在前面，不影响最后的节点embedding数目
        g.G.remove_edges_from(list(test_edges))


    # 模型训练
    model = DeepWalk(graph=g, path_length=args["path_length"],num_paths=args["number_walks"])
    model, embedding, vectors= model.train(args["embedding_size"],  window_size=args["window_size"],
                                   workers = args["workers"],filename=args["output"]+"deepwalk_embedding")
    t2 = time.time()
    print("model train use {} seconds".format(t2-t1))

    # 模型测试
    if args["check_link_pridiction"]:
        check_index = [100, 300, 500, 700, 900, 1100,1300,1500]
        ret = check_link_prediction(embedding, g, origin_graph, check_index)
        print(ret)

    if args["check_reconstruction"]:
        check_index = [100, 300, 500, 700, 900, 1100, 1300,1500]
        ret = check_reconstruction(embedding, g, check_index)
        print(ret)


    # 分类效果测试，有随机扰乱，没有cross-val，增加了split_train_evaluate_cross_val测试！
    if args["label_file"] and args["model"] != 'gcn':
        X, Y = read_node_label(args["label_file"])  #这个是来自classfy的read_node_label, 不同于graph下面的那个！
        print("Training classifier using {:.2f}% nodes...".format(
            args["clf_ratio"]*100))
        clf = Classifier(vectors=vectors, clf=LogisticRegression())
        clf.split_train_evaluate(X, Y, args["clf_ratio"])
        clf.split_train_evaluate_cross_val(X, Y)


if __name__ == "__main__":
    random.seed(1024)
    np.random.seed(1024)
    args = config()
    main(args)