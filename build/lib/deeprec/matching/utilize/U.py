import numpy as np

def node2idx(graph):
    """
    # input: a graph object
    # return: a map(key:index, value: node) and node list
    """
    node2idx = {}
    idx2node = []
    temp = 0
    for node in graph.nodes():
        node2idx[node] = temp
        idx2node.append(node)
        temp += 1
    return idx2node, node2idx


def batch_get(vertices, workers):
    """
    # input: vertices, a dict(key: node, value: the neighboors of the node)
    # return: a sample batch of vertices
    """
    batch_size = (len(vertices) - 1) // workers + 1
    ret = []
    batch_temp = []
    temp = 0
    for v1, nbs in vertices.items():
        batch_temp.append((v1, nbs))
        temp += 1
        if temp % batch_size == 0:
            ret.append(batch_temp)
            batch_temp = []
    if len(batch_temp) > 0:
        ret.append(batch_temp)
    return ret


def batch_num(num, workers):
    """
    # Attention: [] change the number to a list
    # for a example: batch_num(17,5)--->[3,3,3,3,3,2]
    """
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]
    


def create_alias_table(node_pdf):
    """
    ref: https://github.com/shenweichen
    input: a pdf of the node
    """
    
    # trans prob to area
    accept, alias = [0] * len(node_pdf), [0] * len(node_pdf)
    small, large = [], []

    for index, prob in enumerate(node_pdf):
        if prob < 1.0:
            small.append(index)
        else:
            large.append(index)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = node_pdf[small_idx]
        alias[small_idx] = large_idx
        node_pdf[large_idx] = node_pdf[large_idx] - \
            (1 - node_pdf[small_idx])
        if node_pdf[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept, alias


def alias_sample(accept, alias):
    """
    input: accept, alias
    """
    N = len(accept)
    i = int(np.random.random()*N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]