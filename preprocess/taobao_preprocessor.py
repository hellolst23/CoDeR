#!/usr/bin/python3
# -*-coding:utf-8 -*-

"""
preprocess taobao dataset
1) a session is a list ['user_id', 'start_time', 'end_time', 'item_list', 'category_list', 'target_item', 'target_category']
2) item_catgy.pkl: save categories corresponding to item id starting from 1.
"""
"""
数据处理步骤：
1. 删除有缺失值的行(一般没有缺失值)
2. 删除当前items 和 类别 出现次数小于5的 数据（DataFrame 中的rows）
3. 对 user, item, category id重新编码
4. 划分数据，删除长度< 2的 session
5, 划分训练集和测试集，最近的 20% session 作为测试集
6， 训练集和测试集分别存为.pkl文件
"""
#todo
"""
Note: 数据处理可以再加几个步骤
1）item<5 过滤完之后，再过滤一遍session长度小于2
2）test 中的item要出现在训练集合中
3）单session 划分成多个session，在划分完训练集和测试集后
"""


import csv
import pickle
import datetime
import time

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool
from tqdm import tqdm
import os
import json
from collections import Counter


def get_taobao_session(group):
    '''
    After filtering the items and categories less than 5, split sessions again.
    Args:
            group: GroupBy object, the return value of function df.groupby()
    Returns:
            res: a session tuple,(user_id,start_time,end_time,item_list,category_list,target_item,target_category)
    '''
    user_id, df = group
    res = []
    df = df.sort_values('timestamp', ascending=True)  # Sort by time
    time_range = pd.date_range(df['timestamp'].iloc[0],
                               df['timestamp'].iloc[df.shape[0] - 1] + pd.Timedelta(seconds=TIME_INTERVAL),
                               freq='%sS' % TIME_INTERVAL)  # Cut by time
    df.index = df['timestamp']
    for i in range(len(time_range) - 1):
        session_clip = df[time_range[i]: time_range[i + 1]]
        item_list = session_clip['item_id'].tolist()
        category_list = session_clip['category_id'].tolist()
        # If the length of a session is less than MINIMUM_SESSION_LENGTH
        if len(item_list) < MINIMUM_SESSION_LENGTH or len(category_list) < MINIMUM_SESSION_LENGTH:
            continue
        if MAXIMUM_LENGTH < len(item_list):  # cut when reach the maximum length
            item_list = item_list[:MAXIMUM_LENGTH + 1]
            category_list = category_list[:MAXIMUM_LENGTH + 1]

        # Remove the last item                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               em as target
        target_item = item_list.pop(-1)
        target_category = category_list.pop(-1)
        start_time = str(session_clip['timestamp'][0])
        end_time = str(session_clip['timestamp'][session_clip.shape[0] - 1])
        res.append((user_id, start_time, end_time, item_list, category_list, target_item, target_category))
    return res


def filter_items(df_data, minimum_occurrence):
    '''
    delete items and categories that appear less than "minimum_occurrence" times
    Args:
        df_data: original data
        minimum_occurrence: minimum occurrence of a item

    Returns: df_data2: data after filtering
    '''
    item_times = df_data.loc[:, 'item_id'].value_counts()
    cagry_times = df_data.loc[:, 'category_id'].value_counts()
    item_times = item_times[item_times > minimum_occurrence]
    cagry_times = cagry_times[cagry_times > minimum_occurrence]

    df_data = df_data[df_data['item_id'].isin(item_times.index)]
    df_data = df_data[df_data['category_id'].isin(cagry_times.index)]

    return df_data


def split_train_test(sess_list, train_size):
    '''
    split train dataset and test dataset by session start time. The latest 20% sessions are the test dataset.
    Args:
        sess_list: taobao_preprocessor() 的返回值
        [(user_id,start_time,end_time,item_list,category_list,target_item,target_category)]

    Returns: train_list, test_list

    '''
    df = pd.DataFrame(sess_list,
                      columns=['user_id', 'start_time', 'end_time', 'item_list', 'category_list', 'target_item',
                               'target_category'])
    df = df.sort_values('start_time')
    k = int(df.shape[0] * train_size)  # the length of training set
    train = df.iloc[:k].values.tolist()
    test = df.iloc[k:].values.tolist()
    return train, test


def taobao_preprocessor(file_path, save_path, time_interval,minimun_session_length=3, maximum_length=2, minimum_occurrence=5, train_size=0.8,
                        nrows=None, works=1):
    '''
    Load the taobao dataset and preprocess the file into session
    Args:
        file_path: The path of raw Taobao UserBehavior.csv.
        save_path: The path to save the session list and session info, save_path/session.pkl: A list of session, each session is a tuple ('user_id', 'start_time', 'end_time', 'item_list', 'category_list', 'target_item', 'target_category')
        time_interval:  The interval of a session (seconds).
        maximum_length: maximum length of a session.
        minimum_occurrence: minimum occurrence of a item.
        nrows: How many rows of the data to read, default to read all the rows.
        works: How many workers to used. default to 1
    Returns:
    '''
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    session_train_file_path = os.path.join(save_path, 'session_train.pkl')
    session_test_file_path = os.path.join(save_path, 'session_test.pkl')
    session_item_catgy_path = os.path.join(save_path, 'item_catgy.pkl')
    session_info_path = os.path.join(save_path, 'session_info.json')
    data = pd.read_csv(file_path, names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp'],
                       nrows=nrows)
    print("-----------------------Read csv finished.-------------------------@ %ss" % datetime.datetime.now())
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    print(data.isnull().all())  # 判断每一列是否有缺失值
    # data.dropna(axis=1, how='any', inplace=True) #判断一下数据是否有缺失值，有的删掉

    print(Counter(data['behavior_type']))
    data_buy = data[(data['behavior_type'].isin(['buy']))] # NOTE: 新加的 只要buy的数据
    print(Counter(data_buy['behavior_type']))

    # NOTE：根据用户id 随机选取部分数据
    

    # Filter item and category id
    data = filter_items(data, minimum_occurrence)

    # Encoder the user_id, item_id and category_id
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    data['user_id'] = user_encoder.fit_transform(data['user_id'])  # fit_transform()返回array
    data['item_id'] = item_encoder.fit_transform(data['item_id']) + 1  # 编码从1开始，适应sr-gnn的源码，这样后面session补成定长的时候可以区分开来
    data['category_id'] = category_encoder.fit_transform(data['category_id']) + 1
    num_of_user = len(user_encoder.classes_)  # user_encoder.classes_记录未编码前的类别，ndarray
    num_of_item = len(item_encoder.classes_)
    num_of_category = len(category_encoder.classes_)
    # Get category id for all the items
    item_category = data.drop_duplicates(subset=['item_id'], keep='first').sort_values('item_id')['category_id'].tolist()
    session_info = {'num_of_user': num_of_user, 'num_of_item': num_of_item, 'num_of_category': num_of_category}
    assert num_of_item == len(item_category) # , print("Item number is not equal to the item category length")

    # Group by user id
    global TIME_INTERVAL, MAXIMUM_LENGTH,MINIMUM_SESSION_LENGTH
    TIME_INTERVAL = time_interval
    MAXIMUM_LENGTH = maximum_length  # 最长session长度
    MINIMUM_SESSION_LENGTH = minimun_session_length  # 最短session长度
    print('---------------Start splitting sessions--------------------')
    # Multiprocessing
    with Pool(works) as p:
        user_group = data.groupby("user_id")
        session_list = []
        for res in tqdm(p.imap_unordered(get_taobao_session, user_group),
                        total=len(user_group)):
            if res:
                session_list.extend(res)

    print("--------------Total number of session: ", len(session_list))
    session_info['Total number of session'] = len(session_list)
    train_list, test_list = split_train_test(session_list, train_size)
    session_info['Total number of train session'] = len(train_list)
    session_info['Total number of test session'] = len(test_list)
    with open(session_info_path, 'w') as f:
        json.dump(session_info, f)
    with open(session_item_catgy_path, 'wb') as f:
        pickle.dump(item_category, f)
    with open(session_train_file_path, 'wb') as f:
        pickle.dump(train_list, f)

    with open(session_test_file_path, 'wb') as f:
        pickle.dump(test_list, f)
    return


if __name__ == "__main__":
    os.chdir('../')
    time_start = time.time()
    file_load = 'datasets/taobao/UserBehavior.csv'
    saving_load = 'datasets/taobao_11/'
    # file_load = 'datasets/taobao_test/test_sample.csv'
    # saving_load = 'datasets/taobao_test/'
    # taobao_preprocessor('datasets/sample/taobao_sample.csv', 'datasets/sample/', 60 * 60 * 24, works=16)

    taobao_preprocessor(file_load, saving_load, time_interval=60 * 60 * 24,minimun_session_length=5, maximum_length=200, minimum_occurrence=30, train_size=0.8,
                        works=16)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
