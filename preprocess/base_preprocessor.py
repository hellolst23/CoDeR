#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

# @File    : base_preprocessor.py
# Desc:
其他数据集的处理在这个上面展开
处理完后的session的存储格式
1) a session is a list ['user_id', 'start_time', 'end_time', 'item_list', 'category_list', 'target_item', 'target_category']
2) item_catgy.pkl: save categories corresponding to item id starting from 1.
"""
import csv
import pickle
import datetime
import time
import random

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool
from tqdm import tqdm
import os
import json
from collections import Counter
import pickle
# from preprocess import *
from itertools import chain

"""
数据处理步骤：
1. 删除有缺失值的行(一般没有缺失值)
2. 删除当前items 和 类别 出现次数小于5的 数据（DataFrame 中的rows）
3. 对 user, item, category id重新编码
4. 划分数据，删除长度< 2的 session
5. session的数据增强，一个session划成多个session训练与测试
6. 划分训练集和测试集，最近的 20% session 作为测试集
7. 训练集和测试集分别存为.pkl文件
"""



def save_data(save_path,session_info,item_category,train_list,test_list):
    """

    Args:
        save_path: The path to save the session list and session info, save_path/session.pkl: A list of session,
                    each session is a tuple ('user_id', 'start_time', 'end_time', 'item_list', 'category_list', 'target_item', 'target_category')
        session_info,item_category,train_list,test_list: need to be saved
    Returns:

    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    session_train_file_path = os.path.join(save_path, 'session_train.pkl')
    session_test_file_path = os.path.join(save_path, 'session_test.pkl')
    session_item_catgy_path = os.path.join(save_path, 'item_catgy.pkl')
    session_info_path = os.path.join(save_path, 'session_info.json')

    with open(session_info_path, 'w') as f:
        json.dump(session_info, f)
    with open(session_item_catgy_path, 'wb') as f:
        pickle.dump(item_category, f)
    with open(session_train_file_path, 'wb') as f:
        pickle.dump(train_list, f)
    with open(session_test_file_path, 'wb') as f:
        pickle.dump(test_list, f)
    return


def filter_items(df_data, minimum_occurrence):
    """
        delete items and categories that appear less than "minimum_occurrence" times
        Args:
            df_data: original data
            minimum_occurrence: minimum occurrence of a item

        Returns: df_data: data after filtering
    """
    # 已改成与物品在不同时间与用户交互的交互数，这样要求一个物品至少与minimum_occurrence个session中
    df_temp = df_data.loc[:, ['user_id','item_id','category_id', 'timestamp']].drop_duplicates()
    item_times = df_temp.loc[:, 'item_id'].value_counts()
    cagry_times = df_temp.loc[:, 'category_id'].value_counts()
    item_times = item_times[item_times > minimum_occurrence]
    cagry_times = cagry_times[cagry_times > minimum_occurrence]


    df_data = df_data[df_data['item_id'].isin(item_times.index)]
    df_data = df_data[df_data['category_id'].isin(cagry_times.index)]

    return df_data


def split_train_test(sess_list, train_size):
    '''
    split train dataset and test dataset by session start time. The latest 20% sessions (before session data enhancement)are the test dataset.
    Args:
        sess_list: taobao_preprocessor() 的返回值
        [(user_id,start_time,end_time,item_list,category_list,target_item,target_category)]

    Returns: train_list, test_list

    '''
    df = pd.DataFrame(sess_list,
                      columns=['user_id', 'start_time', 'end_time', 'item_list', 'category_list', 'target_item',
                               'target_category'])
    df = df.sort_values('start_time')
    df_temp = df[['user_id', 'start_time']].sort_index()
    print('before len(df_temp): ', len(df_temp))
    df_temp = df_temp.drop_duplicates() # 去掉重复值，来算出没经过session 数据增强之前的session总数
    num_total_session = len(df_temp)
    print('after len(df_temp): ', len(df_temp))
    row_index = df_temp.index.tolist()
    k = int(df_temp.shape[0] * train_size)  # the length of training set
    num_train_sessoion = k
    num_test_session = len(df_temp) - k
    print('k：', k)
    k = row_index[k]
    print('row_index[k]: ',k)
    train = df.iloc[:k].values.tolist()
    test = df.iloc[k:].values.tolist()

    """
    print('---------before exclude_item_notin_training------------')
    print('train: ', train[:3])
    print('test: ', test[:3])
    """
    return train, test, num_train_sessoion, num_test_session, num_total_session


def exclude_item_notin_training(train, test, item_category, session_info):
    """
    1) For test data, exclude items not in training set.
    2) the item id and category id of train data, test data are encoded again.
    Args:
        train: train list, a session:[(user_id,start_time,end_time,item_list,category_list,target_item,target_category)]
        test: test list
        item_category: list, category id corresponding to item id from 1 to n_item

    Returns:
        train_list, test_list
    """
    print('--------------------- code into exclude_item_notin_training --------------------------------')
    df_train = pd.DataFrame(train,
                            columns=['user_id', 'start_time', 'end_time', 'item_list', 'category_list', 'target_item',
                               'target_category'])
    df_test = pd.DataFrame(test, columns=['user_id', 'start_time', 'end_time', 'item_list', 'category_list', 'target_item',
                               'target_category'])

    item_train = list(chain(* df_train['item_list'].values.tolist())) + df_train['target_item'].values.tolist()
    item_train = list(set(item_train)) # 去重复值,
    item_train.sort(reverse=False)  # 按照item id 升序排列
    catgy_train = list(chain(* df_train['category_list'].values.tolist())) + df_train['target_category'].values.tolist()
    catgy_train = list(set(catgy_train))
    catgy_train.sort(reverse=False)
    session_info['num_of_item[not include 0]']=len(item_train)
    session_info['num_of_category'] = len(catgy_train)

    print("session_info['num_of_item[not include 0]']", session_info['num_of_item[not include 0]'])
    print("session_info['num_of_category']", session_info['num_of_category'])

    def merge2session(df):
        """
        merge item_list and target_item into a column session_item
        Args:
            df_test:
        Returns:
        """

        def int2list(item):
            return [item]
        df_target_item = df['target_item'].map(int2list)
        df_sess_i = df['item_list'] + df_target_item
        df_sess_i = pd.DataFrame(df_sess_i, columns=['session_item'])

        df_target_catgy = df['target_category'].map(int2list)
        df_sess_c = df['category_list'] + df_target_catgy
        df_sess_c = pd.DataFrame(df_sess_c, columns=['session_catgy'])

        df_1 = pd.concat([df, df_sess_i, df_sess_c], axis=1)
        return df_1
    df_train_1 = merge2session(df_train)
    df_test_1 = merge2session(df_test)

    def filter_item_test(item_list):
        new_item_list = [item for item in item_list if item in item_train]  # filter_item_test函数外,exclude_item_notin_training函数内的局部变量
        # if item_list != new_item_list:
        #     print('before filtering : ', item_list)
        #     print('after new item_list: ', new_item_list)
        if len(new_item_list) < MINIMUM_SESSION_LENGTH: # 全局变量
            #print('after filtering remove: ', new_item_list)
            new_item_list = None
        return new_item_list
    df_test_1['session_item'] = df_test_1['session_item'].map(filter_item_test)

    def item2catgy(item_list, item_category_list):
        """
            item_category_list: list, category id corresponding to item id from 1 to n_item
        """
        if item_list is None:
            return None
        catgy_list = [item_category_list[item_id-1] for item_id in item_list]
        return catgy_list
    df_test_1['session_catgy'] = df_test_1['session_item'].apply(item2catgy, item_category_list=item_category)
    df_test_1 = df_test_1.dropna()
    print("df_test_1[['session_item', session_catgy']]: \n", df_test_1[['session_item', 'session_catgy']])

    # 重新编码一次
    item_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    item_encoder.fit(item_train)
    category_encoder.fit(catgy_train)

    # 这一段代码与下面注释的两行代码结果一样，因为encoder.fit 是按照item_train中item id 第一次出现的前后顺序来决定编码后的id 的相对大小的
    itemId = item_train
    item_transformed = list(item_encoder.transform(itemId))
    item_item = dict(zip(itemId, item_transformed))
    categoryId = [item_category[i-1] for i in itemId]

    catgy_transformed = list(category_encoder.transform(categoryId))
    item_category_new = dict(zip(item_transformed, catgy_transformed))
    item_category_new_1 = dict(sorted(item_category_new.items(), key=lambda item: item[0]))
    print('item_category_new_1 == item_category_new: ', item_category_new_1 == item_category_new)

    item_category_new_1 = (np.array(list(item_category_new_1.values())) + 1).tolist()  # 前面的encoder 默认编码是从0开始的，这里转成从1开始
    print('type(item_category_new_1[0])', type(item_category_new_1[0]))

    """
    item_category_catgy = [item_category[i-1] for i in item_train ]
    category_new = list(category_encoder.transform(item_category_catgy))
    """

    def session_encode_split(df, item_encoder, category_encoder):
        def transform(session, encoder):
            session_list = encoder.transform(session)
            return session_list + 1
        df['session_item'] = df['session_item'].apply(transform, encoder=item_encoder)
        df['session_catgy'] = df['session_catgy'].apply(transform, encoder=category_encoder)

        df['item_list'] = df['session_item'].map(lambda x: list(x[:-1]))
        df['target_item'] = df['session_item'].map(lambda x: x[-1])
        df['category_list'] = df['session_catgy'].map(lambda x: list(x[:-1]))
        df['target_category'] = df['session_catgy'].map(lambda x: x[-1])
        df = df[['user_id', 'start_time', 'end_time', 'item_list', 'category_list', 'target_item','target_category']]
        return df
    df_test = session_encode_split(df_test_1, item_encoder,category_encoder)
    df_train = session_encode_split(df_train_1, item_encoder, category_encoder)
    return df_train.values.tolist(), df_test.values.tolist(),item_category_new_1, session_info


def get_session(group):
    """
        After filtering the items and categories less than 5, split sessions again.
        Args:
            group: GroupBy object, the return value of function df.groupby()
        Returns:
            res: a session tuple,(user_id,start_time,end_time,item_list,category_list,target_item,target_category)
            or list, [(user_id,start_time,end_time,item_list,category_list,target_item,target_category), ...], if SESSION_ENHANCEMENT is true
    """
    user_id, df = group
    res = []
    df = df.sort_values('timestamp', ascending=True)  # Sort by time
    for timestamp, df_t in df.groupby(by='timestamp'):  # 按天划分session  #todo taobao 数据集在这里不同，但是tafeng和tmall一样
        item_list = df_t['item_id'].tolist()
        category_list = df_t['category_id'].tolist()
        # If the length of a session is less than MINIMUM_SESSION_LENGTH
        if len(item_list) < MINIMUM_SESSION_LENGTH or len(category_list) < MINIMUM_SESSION_LENGTH:
            continue
        if MAXIMUM_LENGTH < len(item_list):  # cut when reach the maximum length
            #print('Long remove: ', item_list[MAXIMUM_LENGTH + 1: ], 'catgy: ',  category_list[MAXIMUM_LENGTH + 1:]) # todo 这里是否考虑要把截断的部分保留
            item_list = item_list[:MAXIMUM_LENGTH + 1]
            category_list = category_list[:MAXIMUM_LENGTH + 1]

        # Remove the last item as target item and split a session into multi-sessions (data enhancement)
        start_time = str(timestamp)
        end_time = str(timestamp)
        if SESSION_ENHANCEMENT:
            for i in range(1, len(item_list) + 2 - MINIMUM_SESSION_LENGTH):
                res.append((user_id, start_time, end_time, item_list[:-i], category_list[:-i], item_list[-i], category_list[-i]))
        else:
            res.append(
                (user_id, start_time, end_time, item_list[:-1], category_list[:-1], item_list[-1], category_list[-1]))
    return res


def base_preprocessor(data, save_path,beizhu='beizhu', sess_enhancement=False,exclude_item=False, minimun_session_length=2, minimum_occurrence=5, time_interval=60 * 60 * 24, maximum_length=50, train_size=0.8, works=1):
    """
    Args:
        data: pandas, columns=['user_id', 'item_id', 'category_id', 'timestamp'], data['timestamp']: dtype= datetime64[ns],such as: 2000-08-29
        save_path: The path to save the session list and session info, save_path/session.pkl: A list of session,
                    each session is a tuple ('user_id', 'start_time', 'end_time', 'item_list', 'category_list', 'target_item', 'target_category')
        sess_enhancement: bool， default=True,
        minimun_session_length: minimum length of a session
        maximum_length: maximum length of a session
        time_interval: The interval of a session (seconds).
        minimum_occurrence: minimum occurrence of a item.
        train_size:
        works: How many workers to used. default to 1
    Returns:
    """
    print(data.isnull().all())# 判断每一列是否有缺失值
    # data.dropna(axis=1, how='any', inplace=True) #判断一下数据是否有缺失值，有的删掉

    # Filter item and category id
    data = filter_items(data, minimum_occurrence)

    # Encoder the user_id, item_id and category_id
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    data['item_id'] = item_encoder.fit_transform(data['item_id']) + 1  # 编码从1开始，适应sr-gnn的源码，这样后面session补成定长的时候可以区分开来
    data['user_id'] = user_encoder.fit_transform(data['user_id'])
    data['category_id'] = category_encoder.fit_transform(data['category_id']) + 1
    num_of_user = len(user_encoder.classes_)  # user_encoder.classes_记录未编码前的类别，ndarray
    num_of_item = len(item_encoder.classes_)
    num_of_category = len(category_encoder.classes_)
    # Get category id for all the items according items id from small to large
    item_category = data.drop_duplicates(subset=['item_id'], keep='first').sort_values('item_id')[
        'category_id'].tolist()  # 只考虑'item_id'列，将这列对应值相同的行进行去重
    session_info = {'beizhu': beizhu,'hyper_minimum_session_length': minimun_session_length,
                    'hyper_maximum_session_length': maximum_length, 'hyper_minimum_occurrence': minimum_occurrence,
                    'hyper_train_size': train_size,
                    'num_of_user': num_of_user, 'num_of_item[not include 0]': num_of_item, 'num_of_category': num_of_category}

    print("session_info['num_of_item[not include 0]']" , session_info['num_of_item[not include 0]'])
    print("session_info['num_of_category']", session_info['num_of_category'])

    try:
        assert num_of_item == len(item_category)
    except:
        print("Item number is not equal to the item category length")

        # Group by user id
    global TIME_INTERVAL, MAXIMUM_LENGTH, MINIMUM_SESSION_LENGTH, SESSION_ENHANCEMENT
    TIME_INTERVAL = time_interval
    MAXIMUM_LENGTH = maximum_length  # 最长session长度
    MINIMUM_SESSION_LENGTH = minimun_session_length  # 最短session长度
    SESSION_ENHANCEMENT = sess_enhancement  # bool,是否进行session data enhancement. Namely, an session is devided into multi sessions
    print('---------------Start splitting sessions--------------------')
    print(' SESSION_ENHANCEMEN: ', SESSION_ENHANCEMENT, '\tMINIMUM_SESSION_LENGTH: ', MINIMUM_SESSION_LENGTH)
    # Multiprocessing
    #"""
    with Pool(works) as p:
        user_group = data.groupby("user_id")
        session_list = []
        for res in tqdm(p.imap_unordered(get_session, user_group),
                        total=len(user_group)):
            if res:
                session_list.extend(res)
    #"""
    # single thread
    """
    user_group = data.groupby("user_id")
    session_list = []
    for user_id, df in tqdm(user_group):
        group = (user_id, df)
        result = get_session(group)
        if result is not None:
            session_list.extend(result)
    #"""
    print("--------------Total number of session: ", len(session_list))
    session_info['Total number of session after enhancement'] = len(session_list)
    train_list, test_list, num_train_sess, num_test_sess, num_total_session = split_train_test(session_list, train_size)
    print('Total number of session before enhancement: ', num_total_session)
    print('num_test_sess', num_test_sess)
    print('num_train_sess: ', num_train_sess)
    print('num_test_sess', num_test_sess)
    session_info['Total number of session before enhancement'] = num_total_session
    session_info['Total number of train session'] = num_train_sess
    session_info['Total number of test session'] = num_test_sess

    if exclude_item:
        train_list, test_list,item_category, session_info = exclude_item_notin_training(train_list, test_list, item_category, session_info)
    print('session_info: ', session_info)
    save_data(save_path=save_path, session_info=session_info, item_category=item_category, train_list=train_list, test_list=test_list)
    return

def load_taobao(file_path, sample_ratio=0.1,nrows=None):
    """
    Load the taobao dataset and preprocess the data format
    Args:
        file_path: The path of raw taobao.csv
        sample_ratio： float, the ratio for sampling user id
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
    # print(Counter(data['behavior_type']))
    # data_buy = data[(data['behavior_type'].isin(['buy','cart','fav']))] # NOTE: 新加的 只要buy的数据
    # print(Counter(data_buy['behavior_type']))
    # print(data_buy.head())
    data.drop('behavior_type',axis=1, inplace=True)

    # NOTE: 只有taobao数据集需要采样
    user_id = data['user_id'].unique()
    print(f"user_id num: {len(user_id)}")
    sample_num = int(0.1*len(user_id))
    print(f"len(user_id):{len(user_id)}")
    sample_list = [i for i in range(len(user_id))]
    sample_user_id = user_id[random.sample(sample_list, sample_num)]
    data_temp = data[data['user_id'].isin(sample_user_id)]
    print(len(data))
    print(len(data_temp))
    return data_temp


def taobao_main():

    os.chdir('../')
    time_start = time.time()
    file_load = 'datasets/taobao/UserBehavior.csv'
    saving_load = 'datasets/taobao_10/'
    # file_load = 'datasets/taobao_test/test_sample.csv'
    # saving_load = 'datasets/taobao_test/'
    # taobao_preprocessor('datasets/sample/taobao_sample.csv', 'datasets/sample/', 60 * 60 * 24, works=16)
    beizhu = ' sample_user_id: 1, sess_enhancement: False'
    data = load_taobao(file_path=file_load, sample_ratio=1, nrows=None)
    base_preprocessor(data, saving_load,beizhu=beizhu, sess_enhancement=False, exclude_item=False, minimun_session_length=2, minimum_occurrence=30,  time_interval= 60 * 60 * 24, maximum_length=50,
                train_size=0.8, works=16)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')

if __name__ == "__main__":
    # #  预处理代码

    # from preprocess.tafeng import load_tafeng
    # os.chdir('../')
    # time_start = time.time()
    # file_load = 'datasets/tafeng/ta_feng.csv'
    # saving_load = 'datasets/tafeng_test/' # session 的长度最小为3，即除去target之后最小是2 #tafeng_singleThread
    # data = load_tafeng(file_path=file_load, nrows=10000)
    # beizhu = 'only 10000 lines for debug, exclude_item=False, sess_enhancement: False '
    # # beizhu = ' sess_enhancement: True, test set not session enhancement, only for train set; \n' \
    # #          'using single thread when preprocessing sessions; \n' \
    # #          'Subsessions belonging to the same session are located together'
    # base_preprocessor(data, saving_load, beizhu=beizhu, sess_enhancement=False, exclude_item=False, minimun_session_length=2,  minimum_occurrence=5, maximum_length=50,time_interval= 60 * 60 * 24, train_size=0.8,
    #                     works=16)
    # time_end = time.time()
    # print('time cost', time_end - time_start, 's')
    
    taobao_main()
