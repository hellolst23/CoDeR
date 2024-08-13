#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    # @File    : util_narm.py
    # Desc:
    some useful functions
"""
import time
import os
import psutil
import torch
from itertools import chain
import numpy as np
import random
from functools import wraps

import torch

#  fix seed
def seed_torch(log_path_txt, seed=None):
    """
    Args:
        log_path_txt: 文件路径，such as： output.txt
        seed:
    Returns: None
    """

    if seed is None:
        seed = int(time.time())
    file_write(log_path_txt, f'************ seed ***********: {seed}')
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子


# 装饰器：实现附加功能，又不影响原函数的结构
def timefn(fn):
    """
    计算函数运行时间的装饰器
    Args:
        fn: a function
    """
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@timefn: {fn.__name__} took {t2 - t1: .5f} s")
        return result
    return measure_time

# 显示当前 python 程序占用的内存大小
def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    print("{} memory used: {} MB".format(hint, memory))


def print_result(log_file, title,metric, Ks=[10,20,50]):
    """
    Args:
        log_file: 文件路径，such as： output.txt
        title: str, 'recall', 'mrr', 'ndcg'
        metric: list, len=len(Ks)
    """
    assert len(metric)==len(Ks)
    title_str = '%s@%d\t' * len(Ks)
    #file_write(log_file, f'{title}@{Ks[0]}\t{title}@{Ks[1]}\t{title}@{Ks[2]}\t{title}@{Ks[3]}\t{title}@{Ks[4]}\t{title}@{Ks[5]}\t{title}@{Ks[6]}')
    file_write(log_file, title_str %(title, Ks[0], title, Ks[1],title, Ks[2],title, Ks[3],title, Ks[4], title, Ks[5],
       title, Ks[6]))
    #file_write(log_file, f"{metric[0]}\t{metric[1]}\t{metric[2]}\t{metric[3]}\t{metric[4]}\t{metric[5]}\t{metric[6]}\t")
    #"""
    result_str = '%.4f\t' * len(Ks)
    file_write(log_file, result_str % (metric[0], metric[1], metric[2], metric[3],metric[4],
        metric[5],metric[6]))
    #"""
    return


def print_auc_result(log_file,title, metric, Ks_auc=[50,100,200,500]):
    """
    Args:
        log_file: 文件路径，such as： output.txt
        title: str, 'auc'
        metric: list, len=len(Ks_auc)
    """
    assert len(metric) == len(Ks_auc)
    """
    file_write(log_file, f'{title}@{Ks_auc[0]}\t{title}@{Ks_auc[1]}\t{title}@{Ks_auc[2]}\t{title}@{Ks_auc[3]}')
    file_write(log_file, f"{metric[0]}\t{metric[1]}\t{metric[2]}\t{metric[3]}")
    """
    #"""
    title_str = '%s@%d\t' * len(Ks_auc)
    file_write(log_file, title_str % (title, Ks_auc[0], title, Ks_auc[1], title, Ks_auc[2], title, Ks_auc[3]))
    result_str = '%.4f\t' * len(Ks_auc)
    file_write(log_file, result_str % (metric[0], metric[1], metric[2], metric[3]))
    #"""
    return


def cprint(log_file, words: str):
    """
    Highlight words
    Args:
        log_file: 文件路径，such as： output.txt
        words: str
    Returns: None
    """
    print(f"\033[31;1m{words}\033[0m")
    file_write(log_file, '\n', whether_print=False)
    file_write(log_file,words, whether_print=False)


def file_write(log_file, s, whether_print=True):
    """
    将字符 s 写入文件 log_file, 以txt形式，非二进制文件
    Args:
        log_file: 文件路径，such as： output.txt
        s: str, 要写入文件的内容
        whether_print: bool， default=True， 是否打印写入的字符串内容
    """
    load_optimal_model = False  # todo 加载最优模型时，需要改成 True， 否则就是False
    if whether_print:
        print(s)
    if not load_optimal_model:
        with open(log_file, 'a') as f:  # 'a' 表示打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。
            f.write(s+'\n')


def sample_negative(sess_nodes_batch, target, item_catgy, n_negative, sample_strategy='category'):
    """
    sample negative examples
    Args:
        sess_nodes_batch: torch.Tensor, dtype=torch.int64,  batch_size * max_nodes_len
        target: torch.Tensor, batch_size, 1维, dtype=torch.long
        item_catgy: torch.Tensor, n_items  include [0], 1维, dtype=torch.long
        n_negative: int, the number of negative samples
        sample_strategy: str, default='random', 'category'
    Returns:
        neg_sample: torch.Tensor, batch_size * n_negative, dtype=torch.long
    """
    batch_size, max_nodes_len = sess_nodes_batch.shape
    n_items = item_catgy.shape[0]
    posindex = []
    item_all = torch.arange(1,n_items).tolist()

    # 随机采样（当前session以及target之外的都做负例）
    for i in torch.arange(batch_size):

        items_byCategory = get_items_byCategory(target[i], item_catgy, sample_strategy=sample_strategy)
        neg_set = list(set(items_byCategory) - set(sess_nodes_batch[i].tolist()) - set([target[i].item()]))
        if len(neg_set) == 0:  # 此时随机选取
            neg_set = list(set(item_all) - set(sess_nodes_batch[i].tolist()) - set([target[i].item()]))

        negitem = [random.choice(neg_set) for j in range(n_negative)]
        posindex.append(negitem)

    return torch.tensor(posindex).long()

def get_items_byCategory(target, item_catgy, sample_strategy=None):
    """
    Obtain items of the same category as the target item.
    Args:
        target: item_id
        item_catgy:  torch.Tensor, n_items  include [0], 1维, dtype=torch.long
        sample_strategy: str, default=None, 'random', 'category'
    Returns:
        items_index: list, items of the same category as the target item
    """
    n_items = item_catgy.shape[0]
    if sample_strategy != 'category':
        return torch.arange(1,n_items).tolist()
    target_category = item_catgy[target]
    item_catgy = item_catgy.cpu().numpy()
    items_index = np.argwhere(item_catgy == target_category.cpu().numpy())  # item_num * 1
    items_index = items_index.reshape(1, -1).tolist()[0]
    return items_index

def normalize_array(arr, type):
    """
        Args:
           arr: one dimension array
           type: str, max_min/z_score/softmax/sigmoid
        Returns:
            norm_arr: normal array
    """
     
    # 找出数组中的最小值和最大值
    min_val = np.min(arr)
    max_val = np.max(arr)
    # 计算数组的均值和标准差
    mean_val = np.mean(arr)
    std_val = np.std(arr)

    arr = arr.astype(float)
    # 求取指数
    exp_arr = np.exp(arr)
    # 相加求和
    sum_arr = np.sum(exp_arr)

    
    if type == 'max_min':
        norm_arr = (arr - min_val) / (max_val - min_val)
    if type == 'z_score':  
        # 对每个元素进行Z-score标准化处理 zscore_arr
        norm_arr = (arr - mean_val) / std_val
    if type == 'softmax':
        # 对每个元素进行softmax归一化处理
        norm_arr = exp_arr / sum_arr
    if type == 'sigmoid':
        # 转换为浮点数后并转换为Sigmoid值
        norm_arr = 1 / (1 + np.exp(-arr))
    
    return norm_arr

def normalize_array_2D(arr, type):
    """
        Args:
           arr: Two dimension array
           type: str, max_min/z_score/softmax
        Returns:
            norm_arr: normal array
    """
     
    # 找出数组中的最小值和最大值
    min_val = np.min(arr)
    max_val = np.max(arr)
    # 计算数组的均值和标准差
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    
    if type == 'max_min':
        norm_arr = (arr - min_val) / (max_val - min_val)
    if type == 'z_score':  
        # 对每个元素进行Z-score标准化处理 zscore_arr
        norm_arr = (arr - mean_val) / std_val
    if type == 'softmax':
        # 对每个元素进行softmax归一化处理
        norm_arr = np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 1, arr)
    
    return norm_arr


if __name__ == '__main__':
    # test pt function

    # test cprint function
    """
    path = 'output.txt'
    a =3
    b = f" ccc {a}"
    cprint(path, f" ccc {a}")
    #"""

    # test sample_negative
    """
    import torch as t
    s = t.LongTensor(t.arange(1, 31, ).view(5, 6))
    n_negative = 3
    n_items=31
    posindex = sample_negative(s, n_negative, n_items)
    print(posindex)
    """

    # 测试代码执行时间
    """
    import torch.nn as nn
    import torch

    @timefn
    def linear(a, m):

        out = m(a)
        print(out.shape)
    @timefn
    def dot(a, w):
        y = torch.matmul(a,w)
        print(y.shape)

    a = torch.randn(10, 100)
    o1 = linear(a, nn.Linear(100, 1))
    o2 = dot(a, nn.Parameter(torch.Tensor(100)))
    """
    # 测试 print_result
    """
    title = 'recall'
    Ks = [10,20,40,50,60,80,100]
    metric = [1,2,3,4,5,6,7]
    print_result(title, metric,Ks)
    """

    # 测试 normalize_array
    # arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # norm_arr = normalize_array(arr, type ='softmax')
    # print(norm_arr)
    