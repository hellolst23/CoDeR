import numpy as np

def node_degree(node,array):
    # 计算节点的度数
    degree =sum(array[node])
    return degree

def A(i,j,array):
    # 判断两个节点是否存在边
    if array[i,j]==0:
        return 0
    else:
        return 1

def k(i,j,array):
    # 计算两个节点的度数积
    kij = node_degree(i,array) *node_degree(j,array)
    return kij

def judge_cluster(i,j,l):
    # 判断两个节点是否在一个社区
    if l[i] == l[j]:
        return 1
    else:
        return 0

def modularity(array,cluster):
    np.seterr(divide='ignore',invalid='ignore')  # 忽略除0警告信息
    q =0
    m =sum(sum(array))/2#总边数
    for i in range(array.shape[0]):
        for j in range(array.shape[0]):
            if judge_cluster(i,j,cluster) != 0:
                q +=(A(i,j,array) - (k(i,j,array)/(2*m))) *judge_cluster(i,j,cluster)
    q = q/(2*m)
    return q

if __name__ == '__main__':

    array = np.array([[0, 0, 1],
                      [0, 0, 1],
                      [0, 1, 0]])
    cluster = [1, 1, 2]
    print(modularity(array, cluster))