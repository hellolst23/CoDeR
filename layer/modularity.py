import numpy as np
def node_degree(node,array):
    degree =sum(array[node])
    return degree

def A(i,j,array):
    if array[i,j]==0:
        return 0
    else:
        return 1

def k(i,j,array):
    kij = node_degree(i,array) * node_degree(j,array)
    return kij

def judge_cluster(i,j,l):
    if l[i] == l[j]:
        return 1
    else:
        return 0

def modularity(array,cluster):
    q = 0
    m =sum(sum(array))/2
    for i in range(array.shape[0]):
        for j in range(array.shape[0]):
            q += (A(i,j,array) - (k(i,j,array)/(2*m))) * judge_cluster(i,j,cluster)
    q = q/(2*m)
    return q

def get_modularity(adj_matrixes, session_nodes):
    node_modularity = np.zeros(len(adj_matrixes), dtype=float)
    np.seterr(divide='ignore',invalid='ignore')
    for userID in range(len(adj_matrixes)):
        node_modularity[userID] = modularity(np.nan_to_num(adj_matrixes[userID]), np.nan_to_num(session_nodes[userID]))
    node_modularity = np.nan_to_num(node_modularity)
    
    return node_modularity
