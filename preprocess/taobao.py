#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : taobao.py
# Desc:
"""
# from preprocess import *
from base_preprocessor import base_preprocessor
import pickle

def time_process(time):

    time = str(time)[:-2]+'/'+str(time)[-2:]+'/'+'2000'
    return time


def load_taobao(file_path, nrows=None):
    """
    Load the taobao dataset and preprocess the data format
    Args:
        file_path: The path of raw Tmall.csv
    Returns:
        data：pandas, columns=['user_id', 'item_id', 'category_id', 'timestamp'], data['timestamp']: dtype= datetime64[ns], eg：2000-08-29
    """
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # session_train_file_path = os.path.join(save_path, 'session_train.pkl')
    # session_test_file_path = os.path.join(save_path, 'session_test.pkl')
    # session_item_catgy_path = os.path.join(save_path, 'item_catgy.pkl')
    # session_info_path = os.path.join(save_path, 'session_info.json')
    print("-----------------------Start reading csv -------------------------@ %ss" % datetime.datetime.now())
    data = pd.read_csv(file_path, names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp'],
                       nrows=nrows)
    print("-----------------------Read csv finished.-------------------------@ %ss" % datetime.datetime.now())
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    print(Counter(data['behavior_type']))
    data_buy = data[(data['behavior_type'].isin(['buy']))] 
    print(Counter(data_buy['behavior_type']))
    print(data_buy.head())
    data = data_buy['user_id', 'item_id', 'category_id', 'timestamp']
    return data


def taobao_main():

    os.chdir('../')
    time_start = time.time()
    #file_load = 'datasets/taobao/UserBehavior.csv'
    #saving_load = 'datasets/taobao/'
    file_load = 'datasets/taobao_test/test_sample.csv'
    saving_load = 'datasets/taobao_test/'
    # taobao_preprocessor('datasets/sample/taobao_sample.csv', 'datasets/sample/', 60 * 60 * 24, works=16)
    beizhu = ' sess_enhancement: False'
    data = load_taobao(file_path=file_load, nrows=None)
    base_preprocessor(data, saving_load,beizhu=beizhu, sess_enhancement=False, exclude_item=False, minimun_session_length=5, minimum_occurrence=30,  time_interval= 60 * 60 * 24, maximum_length=50,
               train_size=0.8, works=16)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')


if __name__ == "__main__":

    tmall_main()
    # #"""
    # saving_dir = '../datasets/tmall_minoccur10_0321/'
    # following_preprocessor(saving_dir) 
    # #"""
