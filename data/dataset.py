#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

# @File    : dataset.py
# Desc:
"""
import os
import pickle
import numpy as np
import torch
from torch.utils import data
from sklearn.model_selection import train_test_split
import pandas as pd
import json

from util.utils import file_write


def _following_preprocessor(raw_dataset):
    title = ['user_id', 'start_time', 'end_time', 'item_list', 'category_list', 'target_item', 'target_category']
    df = pd.DataFrame(raw_dataset, columns=title)
    sess_list_i = df['item_list'].values.tolist()
    sess_list_c = df['category_list'].values.tolist()
    tar_i = df['target_item'].values.tolist()
    tar_c = df['target_category'].values.tolist()
    return (sess_list_i, tar_i), (sess_list_c, tar_c)


def load_data(dataset_dir, validation_flag=True, valid_portion=0.5):
    '''
    1) Loads the dataset
    2）划分训练集，验证集
    Args:
        dataset_dir: eg.'datasets/tb_sample'
        valid_portion:
    Returns:
        train, valid, test: ([session_list_0, ... ],[target_0,...])
        item_category: torch.tensor, n_item + 1, [0, category_id, ...]
    '''

    # Load the dataset
    path_train_data = os.path.join(dataset_dir, 'session_train.pkl')
    path_test_data = os.path.join(dataset_dir, 'session_test.pkl')
    path_item2category_path = os.path.join(dataset_dir, 'item_catgy.pkl')
    session_info_path = os.path.join(dataset_dir, 'session_info.json')
    with open(session_info_path, 'r') as f:
        session_info = json.load(f)
    with open(path_train_data, 'rb') as f1:
        train_set = pickle.load(f1)
    with open(path_test_data, 'rb') as f2:
        test_set = pickle.load(f2)
    with open(path_item2category_path, 'rb') as f3:
        item_category = pickle.load(f3)

    valid_set = test_set
    if validation_flag:
        valid_set, test_set = train_test_split(test_set, test_size=valid_portion, random_state=123)
    item_category.insert(0, 0)  # add empty category at font 
    item_category = torch.tensor(item_category).long()
    return train_set, valid_set, test_set, item_category, session_info


def process_adj_matrix(solid_adj_matrix):
    '''
    对原始邻接矩阵进行进一步加工处理
    Args:
        solid_adj_matrix: max_nodes_len*max_nodes_len
    Returns:
        solid_adj_matrix

    '''
    return solid_adj_matrix


def sess_collate_fn(batch):
    '''
    1）一个batch 数据, 定长处理
        i）sess_categories 处理成定长 max_session_len （一个batch中，session长度的最大值）
        ii）sess_items 后相异节点的个数 处理成定长 max_nodes_len（一个batch中，session不同节点数的最大值）
    2）获取sess_items 对应的一阶邻接矩阵 adj_matrix 维度：max_nodes_len*max_nodes_len
    3) 获取 session中相异节点与其sess_categories 的一一对应矩阵（映射关系） nodes_categories_matrix 维度：max_nodes_len*max_session_len
    Args:
        batch: a list, [((sess_items, sess_categories), (target_item, target_item_category)), ...  ], len = batch_size
    Returns:
        sess_nodes_batch： torch.Tensor, dtype=torch.int64,  batch_size * max_nodes_len
        sess_categories_batch: torch.Tensor， dtype = torch.int64, batch_size * max_session_len
        adj_matrix_batch： torch.Tensor, dtype=torch.float64, batch_size *  max_nodes_len* max_nodes_len
        nodes_categories_matrixes: torch.Tensor, dtype=torch.float64, batch_size * max_nodes_len* max_session_len
        sess_target_batch: torch.Tensor, dtype=torch.int64, batch_size * 2 , [[target_item_id, target_categories_id],...]
        session_last_item_index_batch: torch.Tensor, dtype=torch.int64, batch_size, record the last item index in unique nodes
        mask_batch: torch.Tensor, dtype=torch.int64, batch_size * max_nodes_len, record the clicked nodes in a session
    '''
    sess_categories_batch = []
    sess_target_batch = []
    sess_nodes_batch = []
    adj_matrix_batch = []
    nodes_categories_matrix_batch = []
    mask_item_batch = [] # 存 unique nodes 序列的 mask矩阵
    mask_catgy_batch = []  # 存 catgy 序列的 mask矩阵
    session_last_item_index_batch = []
    session_last_catgy_index_batch = []
    num_nodes = []  # 存一个session相异节点数
    num_sess_batch = []  # 存一个session的长度

    for sess in batch:
        sess_items = sess[0][0]
        num_sess_batch.append(len(sess_items))
        num_nodes.append(len(np.unique(sess_items)))
    max_session_len = np.max(num_sess_batch)
    max_nodes_len = np.max(num_nodes)

    for session, target in batch:
        sess_target_batch.append(target)
        # 补成定长
        sess_catgories = session[1]
        sess_categories_batch.append(sess_catgories + ((max_session_len - len(sess_catgories)) * [0]))
        session_last_catgy_index_batch.append(len(sess_catgories) - 1)
        sess_items = session[0]
        last_item = sess_items[-1]  # Get last item
        u, ind = np.unique(sess_items, return_index=True)
        unique_items = u[np.argsort(ind)]  # Sort by occurrence position
        sess_nodes_batch.append(unique_items.tolist() + ((max_nodes_len - len(unique_items)) * [0]))
        session_last_item_index_batch.append((np.where(unique_items == last_item))[0][0])  # Get the index in unique_items

        # todo: 加last node index
        # 获取邻接矩阵, not self-loop, namely, a_ii != 1, unless (v_i, v_i)appears
        # 获取相异节点与其sess_categories 的映射矩阵 nodes_categories_matrix
        # 获取mask矩阵
        solid_adj_matrix = np.zeros((max_nodes_len, max_nodes_len))
        nodes_categories_matrix = np.zeros((max_nodes_len, max_session_len))
        mask = np.zeros(max_nodes_len)
        mask_catgy = np.zeros(max_session_len)
        sess_len = len(sess_items)  # length of a session
        mask_catgy[sess_len - 1] = 1
        for i in np.arange(sess_len - 1):
            mask_catgy[i] = 1
            u = np.where(unique_items == sess_items[i])[0][
                0]  # np.where() 返回一个元组， 元组元素只有一个，为一个array数组， 记录node array数组中值为session item id 的索引
            if not mask[u]: 
                mask[u] = 1
            v = np.where(unique_items == sess_items[i + 1])[0][0]
            if not solid_adj_matrix[u][v]:
                solid_adj_matrix[u][v] = 1  # TODO: 是否要考虑多种边的情况
            if not nodes_categories_matrix[u][i]:
                nodes_categories_matrix[u][i] = 1 # 每一行 sum 不一定等于1， 等于该node在session中出现的次数

        solid_adj_matrix = process_adj_matrix(solid_adj_matrix)
        adj_matrix_batch.append(solid_adj_matrix)

        u_last = np.where(unique_items == sess_items[sess_len-1])[0][0]
        nodes_categories_matrix[u_last][sess_len-1] = 1
        nodes_categories_matrix_batch.append(nodes_categories_matrix)
        if not mask[u_last]:
            mask[u_last] = 1
        mask_item_batch.append(mask)
        mask_catgy_batch.append(mask_catgy)

    return torch.tensor(sess_nodes_batch).long(), torch.tensor(sess_categories_batch).long(), torch.tensor(
        adj_matrix_batch).long(), torch.tensor(nodes_categories_matrix_batch).long(), torch.tensor(
        sess_target_batch).long(), torch.tensor(session_last_item_index_batch).long(), torch.tensor(mask_item_batch).long(),torch.tensor(session_last_catgy_index_batch).long(), torch.tensor(mask_catgy_batch).long()


class RsData(data.Dataset):
    '''
    define the pytorch Dataset class for taobao datasets.
    '''

    def __init__(self, log_dir_txt, data, sliding_size=2):
        self.data = data
        self.sliding_size = sliding_size

        # file_write(log_dir_txt, '-' * 50)
        # file_write(log_dir_txt, 'Dataset info:')
        # file_write(log_dir_txt, 'Number of sessions: {}'.format(len(data)))
        # file_write(log_dir_txt, '-' * 50)

    def __getitem__(self, index):
        user_id, start_time, end_time, item_list, category_list, target_item, target_category = self.data[index]
        return (item_list, category_list), (target_item, target_category)

    def __len__(self):
        return len(self.data)

    def get_adj_matrix(self, sess_items, sliding_size=2):
        '''
        获取一个session的邻接矩阵
        Args:
            sess_items: [item_id_0, ... ], 一个session序列

        Returns:
            adj_matrix：(solid_adj_matrix, dashed_adj_matrix), dtype = np.array
            the adjacency matrix of a session, 索引按照该session的item_id 从小到大的顺序排列, 实线连接(有向，只算了入度的)，和虚线连接的分别用一个矩阵存起来
            example： sess_items = [6,3,4,3,1,5]
            solid_adj_matrix = [[0,0,0,0,1],
                                [1,0,1,0,0]，
                                [0,1,0,0,0],
                                [0,0,0,0,0],
                                [0,1,0,0,0]]
            dashed_adj_matrix = [[0,0,0,0,0], #sliding_size = 3
                                [1,1,0,1,0]，
                                [1,0,0,1,0],
                                [0,0,0,0,0],
                                [0,1,1,0,0]]
        '''
        unique_items = np.unique(sess_items)
        m_len = len(unique_items)
        solid_adj_matrix = np.zeros((m_len, m_len))
        dashed_adj_matrix = np.zeros((m_len, m_len))
        for i in np.arange(len(sess_items) - 1):  # TODO 原始邻接矩阵，没有做其他处理
            u = np.where(unique_items == sess_items[i])[0][
                0]  # np.where() 返回一个元组， 元组元素只有一个，为一个array数组， 记录node array数组中值为session item id 的索引
            v = np.where(unique_items == sess_items[i + 1])[0][0]
            solid_adj_matrix[u][v] += 1
            if sliding_size < 2:  # TODO 原始虚线邻接矩阵，按滑动窗口，没有做其他处理
                raise Exception('sliding_size < 2, there are not dashed edges ')
            else:
                if i <= len(sess_items) - 1 - sliding_size:
                    for k in np.arange(2, sliding_size + 1):
                        v_1 = np.where(unique_items == sess_items[i + k])[0][0]
                        dashed_adj_matrix[u][v_1] += 1
                else:
                    pass
        return solid_adj_matrix, dashed_adj_matrix
