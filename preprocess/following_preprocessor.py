#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @File    : following_preprocessor.py



import os
import pickle
import pandas as pd
import numpy as np
import json

def fun(df):
    '''
    Args:
        df: DataFrame

    Returns:

    '''
    print(df.head())
    sess_list_i = df['item_list'].values.tolist()
    sess_list_c = df['category_list'].values.tolist()
    tar_i = df['target_item'].values.tolist()
    tar_c = df['target_category'].values.tolist()
    print('sess_list_i[:10]', sess_list_i[:10])
    print('sess_list_c[:10]', sess_list_c[:10])
    print('tar_i', tar_i[:10])
    print('tar_c', tar_c[:10])
    return (sess_list_i, tar_i), (sess_list_c, tar_c)

def following_preprocessor(dir):

    print('------------------start reading------------------')
    tra_path = os.path.join(dir, 'session_train.pkl')
    tes_path = os.path.join(dir, 'session_test.pkl')
    sess_info_path = os.path.join(dir, 'session_info.json')
    title = ['user_id', 'start_time', 'end_time', 'item_list', 'category_list', 'target_item', 'target_category']
    with open(tra_path, 'rb') as f1:
        tra_list = pickle.load(f1)
    with open(tes_path, 'rb') as f2:
        tes_list = pickle.load(f2)
    with open(os.path.join(dir, 'item_catgy.pkl'), 'rb') as f3:
        items_catgies = pickle.load(f3) # list
    with open(sess_info_path, 'r') as f4:
        sess_info = json.loads(f4.read())
    n_item = sess_info['num_of_item[not include 0]']
    items_catgies.insert(0,0)
    tra_df = pd.DataFrame(tra_list, columns=title)
    tes_df = pd.DataFrame(tes_list, columns=title)
    train, train_c = fun(tra_df)
    test, test_c = fun(tes_df)
    items_catgies_list = [np.arange(n_item+1).tolist(), items_catgies]

    print('------------------start saving------------------')
    with open(os.path.join(dir, 'item_catgy.txt'), 'wb') as f:
        pickle.dump(items_catgies_list, f)
    with open(os.path.join(dir, 'train.txt'), 'wb') as f:
        pickle.dump(train, f)
    with open(os.path.join(dir, 'train_c.txt'), 'wb') as f:
        pickle.dump(train_c, f)
    with open(os.path.join(dir, 'test.txt'), 'wb') as f:
        pickle.dump(test,f)
    with open(os.path.join(dir, 'test_c.txt'), 'wb') as f:
        pickle.dump(test_c, f)
    print('------------------finish saving------------------')
    return None


if __name__ == '__main__':
    dir = '../datasets/taobao_01'
    #dir = '../datasets/tafeng_2'
    train_file_path = os.path.join(dir,'session_train.pkl')
    test_file_path = os.path.join(dir, 'session_test.pkl')
    following_preprocessor(dir)
    print('done')
