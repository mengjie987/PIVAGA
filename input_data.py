import networkx as nx
import scipy.sparse as sp
import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix

def load_data():
    # Load datasets from text files
    def file2array(path, delimiter=' '):  # delimiter是数据分隔符
        fp = open(r'/home/fumengjie/DTI/3.14/data/KEGGGraphs2.txt', 'r', encoding='utf-8')
        string = fp.read()  # string是一行字符串，该字符串包含文件所有内容
        fp.close()
        row_list = string.splitlines()  # splitlines默认参数是‘\n’
        data_list = [[int(i) for i in row.strip().split(delimiter)] for row in row_list]
        return np.array(data_list)

    data = file2array(r'./KEGGGraphs2.txt')
    adj = csr_matrix(data)
    # Transpose the adjacency matrix, as Cora raw dataset comes with a
    # <ID of cited paper> <ID of citing paper> edgelist format.
    adj = adj.T
    features = sp.identity(adj.shape[0])
    return adj, features

