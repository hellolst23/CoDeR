#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

# @File    : tmall.py
# Desc:
处理天猫数据
"""
from preprocess import *
from preprocess.base_preprocessor import base_preprocessor
import pickle

def time_process(time):
    """
    将time（mmdd）转化成 （mm/dd/yyyy）形式
    Args:
        time: timestamp in tmall dataset, "mmdd"

    Returns: time:  "dd/mm/yyyy "的形式
    """
    time = str(time)[:-2]+'/'+str(time)[-2:]+'/'+'2000'
    return time


def load_tmall(file_path, nrows=None):
    """
    Load the tamll dataset and preprocess the data format
    Args:
        file_path: The path of raw Tmall.csv
    Returns:
        data：pandas, columns=['user_id', 'item_id', 'category_id', 'timestamp'], data['timestamp']: dtype= datetime64[ns], eg：2000-08-29
    """
    print("-----------------------Start reading csv -------------------------@ %ss" % datetime.datetime.now())
    data = pd.read_csv(file_path, nrows=nrows)
    print("-----------------------Finish reading csv-------------------------@ %ss" % datetime.datetime.now())
    data = data[['user_id', 'item_id', 'cat_id', 'time_stamp', 'action_type']]
    data.rename(columns={'cat_id': "category_id", 'time_stamp': "timestamp"}, inplace=True)
    data = data.loc[data['action_type'] == 2]  # only retain buy records
    data['timestamp'] = data['timestamp'].apply(time_process)  # "dd/mm/yyyy "的形式
    data['timestamp'] = pd.to_datetime(data['timestamp'],
                                       infer_datetime_format=True)  # dtype= datetime64[ns], eg：2000-08-29
    return data


def tmall_main():
    #  预处理代码
    os.chdir('../')
    time_start = time.time()
    file_load = 'datasets/tmall/tmall.csv'
    saving_load = 'datasets/tmall_minoccur30_0321/'  # session 的长度最小为3，即除去target之后最小是2
    beizhu = ' sess_enhancement: False'
    data = load_tmall(file_path=file_load, nrows=None)
    base_preprocessor(data, saving_load,beizhu=beizhu, sess_enhancement=False, exclude_item=False, minimun_session_length=5, minimum_occurrence=30,  time_interval= 60 * 60 * 24, maximum_length=50,
               train_size=0.8, works=16)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')

if __name__ == "__main__":

    #tmall_main()
    #"""
    saving_dir = '../datasets/tmall_minoccur10_0321/'
    following_preprocessor(saving_dir)  # 转成SR-GNN 的接口
    #"""
