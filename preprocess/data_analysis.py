#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

# @File    : data_analysis.py
# Desc:
分析数据集的各个特性
"""
import os
import pickle
import json

import pandas as pd
import numpy as np

def load_data(dir):
    """
        Args:
            dir: 读取文件的根目录
        Returns:
    """
    print('------------------ start reading ------------------')
    tra_path = os.path.join(dir, 'session_train.pkl')
    tes_path = os.path.join(dir, 'session_test.pkl')
    sess_info_path = os.path.join(dir, 'session_info.json')
    # title = ['user_id', 'start_time', 'end_time', 'item_list', 'category_list', 'target_item', 'target_category']
    with open(tra_path, 'rb') as f1:
        tra_list = pickle.load(f1)
    with open(tes_path, 'rb') as f2:
        tes_list = pickle.load(f2)
    with open(os.path.join(dir, 'item_catgy.pkl'), 'rb') as f3:
        items_catgies = pickle.load(f3)  # list
    with open(sess_info_path, 'r') as f4:
        sess_info = json.loads(f4.read())
    print('------------------ Read Successfully ------------------')
    return tra_list, tes_list, items_catgies, sess_info

def data_analysis(tra_list, tes_list, items_catgies, sess_info):
    """
    输入为load_data的输出
    Args:
        tra_list:
        tes_list:
        items_catgies:
        sess_info:

    Returns:

    """
    title = ['user_id', 'start_time', 'end_time', 'item_list', 'category_list', 'target_item', 'target_category']
    # print(tra_list[:30])
    tra_df = pd.DataFrame(tra_list, columns=title)
    tes_df = pd.DataFrame(tes_list, columns=title)

    view = tes_df[['item_list', 'category_list', 'target_item', 'target_category']]
    cnt = 0
    for i in range(0, len(view)):
        if tes_df['target_item'][i] in tes_df['item_list'][i]:
            cnt += 1
    print(f"cnt: {cnt}")
    print(f"portion: {cnt/len(view)}")

    print(view[0:30])
    print(tra_df['item_list'][100:130])
    print(tra_df['category_list'][100:130])
    print('len(tra_df): ', len(tra_df))
    temp = tra_df['item_list'].tolist()
    a = [len(i) for i in temp]
    session_avg = np.mean(a)
    print(f"session_avg: {session_avg}")


def load_data_txt(dir):
    """
        Args:
            dir: 读取文件的根目录
        Returns:
    """
    print('------------------ start reading ------------------')
    tra_path = os.path.join(dir, 'train.txt')
    tra_path_c = os.path.join(dir, 'train_c.txt')
    tes_path = os.path.join(dir, 'test.txt')
    tes_path_c = os.path.join(dir, 'test_c.txt')
    sess_info_path = os.path.join(dir, 'session_info.json')
    # title = ['user_id', 'start_time', 'end_time', 'item_list', 'category_list', 'target_item', 'target_category']
    with open(tra_path, 'rb') as f1:
        tra_list_txt = pickle.load(f1)
    with open(tra_path_c, 'rb') as f1:
        tra_list_txt_c = pickle.load(f1)
    with open(tes_path, 'rb') as f2:
        tes_list_txt = pickle.load(f2)
    with open(tes_path_c, 'rb') as f2:
        tes_list_txt_c = pickle.load(f2)
    with open(sess_info_path, 'r') as f4:
        sess_info = json.loads(f4.read())
    print('------------------ Read Successfully ------------------')
    return tra_list_txt,tra_list_txt_c, tes_list_txt,tes_list_txt_c, sess_info

def data_analysis_txt(tra_list_txt, tra_list_txt_c, tes_list_txt, tes_list_txt_c, sess_info):
    title = ['item_list', 'target_item']
    print(type(tra_list_txt))





if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth', 100)
    dir = '../datasets/taobao_01'#tmall_minoccur30_0321'
    tra_list, tes_list, items_catgies, sess_info = load_data(dir)
    data_analysis(tra_list, tes_list, items_catgies, sess_info)
    # tra_list_txt, tra_list_txt_c, tes_list_txt, tes_list_txt_c, sess_info = load_data_txt(dir)
    # data_analysis_txt(tra_list_txt, tra_list_txt_c, tes_list_txt, tes_list_txt_c, sess_info)
