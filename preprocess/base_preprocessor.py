import os
import json
import pickle
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain
from multiprocessing import Pool
from sklearn.preprocessing import LabelEncoder

def save_data(save_path,session_info,category,train_list,test_list,item_category):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    session_train_file_path = os.path.join(save_path, 'session_train.pkl')
    session_test_file_path = os.path.join(save_path, 'session_test.pkl')
    session_catgy_path = os.path.join(save_path, 'item_catgy.pkl')
    session_info_path = os.path.join(save_path, 'session_info.json')
    session_item_catgy_path = os.path.join(save_path, 'item_category.pkl')

    with open(session_info_path, 'w') as f:
        json.dump(session_info, f)
    with open(session_catgy_path, 'wb') as f:
        pickle.dump(category, f)
    with open(session_train_file_path, 'wb') as f:
        pickle.dump(train_list, f)
    with open(session_test_file_path, 'wb') as f:
        pickle.dump(test_list, f)
    with open(session_item_catgy_path, 'wb') as f:
        pickle.dump(item_category, f)
    return


def filter_items(df_data, minimum_occurrence):
    df_temp = df_data.loc[:, ['user_id','item_id','category_id', 'timestamp']].drop_duplicates()
    item_times = df_temp.loc[:, 'item_id'].value_counts()
    cagry_times = df_temp.loc[:, 'category_id'].value_counts()
    item_times = item_times[item_times > minimum_occurrence]
    cagry_times = cagry_times[cagry_times > minimum_occurrence]

    df_data = df_data[df_data['item_id'].isin(item_times.index)]
    df_data = df_data[df_data['category_id'].isin(cagry_times.index)]

    return df_data


def split_train_test(sess_list, train_size):
    df = pd.DataFrame(sess_list,
                      columns=['user_id', 'start_time', 'end_time', 'item_list', 'category_list', 'target_item',
                               'target_category'])
    df = df.sort_values('start_time')
    df_temp = df[['user_id', 'start_time']].sort_index()
    print('before len(df_temp): ', len(df_temp))
    df_temp = df_temp.drop_duplicates()
    num_total_session = len(df_temp)
    print('after len(df_temp): ', len(df_temp))
    row_index = df_temp.index.tolist()
    k = int(df_temp.shape[0] * train_size)
    num_train_sessoion = k
    num_test_session = len(df_temp) - k
    print('k:', k)
    k = row_index[k]
    print('row_index[k]: ',k)
    train = df.iloc[:k].values.tolist()
    test = df.iloc[k:].values.tolist()
    return train, test, num_train_sessoion, num_test_session, num_total_session


def exclude_item_notin_training(train, test, item_category, session_info):
    print('--------------------- code into exclude_item_notin_training --------------------------------')
    df_train = pd.DataFrame(train,
                            columns=['user_id', 'start_time', 'end_time', 'item_list', 'category_list', 'target_item',
                               'target_category'])
    df_test = pd.DataFrame(test, columns=['user_id', 'start_time', 'end_time', 'item_list', 'category_list', 'target_item',
                               'target_category'])

    item_train = list(chain(* df_train['item_list'].values.tolist())) + df_train['target_item'].values.tolist()
    item_train = list(set(item_train))
    item_train.sort(reverse=False)
    catgy_train = list(chain(* df_train['category_list'].values.tolist())) + df_train['target_category'].values.tolist()
    catgy_train = list(set(catgy_train))
    catgy_train.sort(reverse=False)
    session_info['num_of_item[not include 0]']=len(item_train)
    session_info['num_of_category'] = len(catgy_train)

    print("session_info['num_of_item[not include 0]']", session_info['num_of_item[not include 0]'])
    print("session_info['num_of_category']", session_info['num_of_category'])

    def merge2session(df):
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
        new_item_list = [item for item in item_list if item in item_train]
        if len(new_item_list) < MINIMUM_SESSION_LENGTH:
            new_item_list = None
        return new_item_list
    df_test_1['session_item'] = df_test_1['session_item'].map(filter_item_test)

    def item2catgy(item_list, item_category_list):
        if item_list is None:
            return None
        catgy_list = [item_category_list[item_id-1] for item_id in item_list]
        return catgy_list
    df_test_1['session_catgy'] = df_test_1['session_item'].apply(item2catgy, item_category_list=item_category)
    df_test_1 = df_test_1.dropna()
    print("df_test_1[['session_item', session_catgy']]: \n", df_test_1[['session_item', 'session_catgy']])

    item_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    item_encoder.fit(item_train)
    category_encoder.fit(catgy_train)

    itemId = item_train
    item_transformed = list(item_encoder.transform(itemId))
    item_item = dict(zip(itemId, item_transformed))
    categoryId = [item_category[i-1] for i in itemId]

    catgy_transformed = list(category_encoder.transform(categoryId))
    item_category_new = dict(zip(item_transformed, catgy_transformed))
    item_category_new_1 = dict(sorted(item_category_new.items(), key=lambda item: item[0]))
    print('item_category_new_1 == item_category_new: ', item_category_new_1 == item_category_new)

    item_category_new_1 = (np.array(list(item_category_new_1.values())) + 1).tolist()
    print('type(item_category_new_1[0])', type(item_category_new_1[0]))

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
    df_train = session_encode_split(df_train_1, item_encoder, category_encoder) #todo 能跑通，回来检查一下是不是对的
    return df_train.values.tolist(), df_test.values.tolist(),item_category_new_1, session_info


def get_session(group):
    user_id, df = group
    res = []
    df = df.sort_values('timestamp', ascending=True) 
    for timestamp, df_t in df.groupby(by='timestamp'):
        item_list = df_t['item_id'].tolist()
        category_list = df_t['category_id'].tolist()
        if len(item_list) < MINIMUM_SESSION_LENGTH or len(category_list) < MINIMUM_SESSION_LENGTH:
            continue
        if MAXIMUM_LENGTH < len(item_list):
            item_list = item_list[:MAXIMUM_LENGTH + 1]
            category_list = category_list[:MAXIMUM_LENGTH + 1]

        start_time = str(timestamp)
        end_time = str(timestamp)
        if SESSION_ENHANCEMENT:
            for i in range(1, len(item_list) + 2 - MINIMUM_SESSION_LENGTH):
                res.append((user_id, start_time, end_time, item_list[:-i], category_list[:-i], item_list[-i], category_list[-i]))
        else:
            res.append(
                (user_id, start_time, end_time, item_list[:-1], category_list[:-1], item_list[-1], category_list[-1]))
    return res


def base_preprocessor(data, save_path,beizhu='beizhu', sess_enhancement=False, exclude_item=False, minimun_session_length=5, minimum_occurrence=30, time_interval=60 * 60 * 24, maximum_length=50, train_size=0.8, works=1):
    print(data.isnull().all())
    data = filter_items(data, minimum_occurrence)

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    data['item_id'] = item_encoder.fit_transform(data['item_id']) + 1
    data['user_id'] = user_encoder.fit_transform(data['user_id'])
    data['category_id'] = category_encoder.fit_transform(data['category_id']) + 1
    num_of_user = len(user_encoder.classes_)
    num_of_item = len(item_encoder.classes_)
    num_of_category = len(category_encoder.classes_)
    category = data.drop_duplicates(subset=['item_id'], keep='first').sort_values('item_id')[
        'category_id'].tolist()
    session_info = {'beizhu': beizhu,'hyper_minimum_session_length': minimun_session_length,
                    'hyper_maximum_session_length': maximum_length, 'hyper_minimum_occurrence': minimum_occurrence,
                    'hyper_train_size': train_size,
                    'num_of_user': num_of_user, 'num_of_item[not include 0]': num_of_item, 'num_of_category': num_of_category}
    item_category = {}
    for i in range(len(data)):
        if data.iloc[i, 2] not in item_category:
            item_category[data.iloc[i, 2]] = [data.iloc[i, 3]]
    print("session_info['num_of_item[not include 0]']" , session_info['num_of_item[not include 0]'])
    print("session_info['num_of_category']", session_info['num_of_category'])

    try:
        assert num_of_item == len(category)
        print("Item number is equal to the item category length")
    except:
        print("Item number is not equal to the item category length")

    global TIME_INTERVAL, MAXIMUM_LENGTH, MINIMUM_SESSION_LENGTH, SESSION_ENHANCEMENT
    TIME_INTERVAL = time_interval
    MAXIMUM_LENGTH = maximum_length
    MINIMUM_SESSION_LENGTH = minimun_session_length
    SESSION_ENHANCEMENT = sess_enhancement
    print('---------------Start splitting sessions--------------------')
    print(' SESSION_ENHANCEMEN: ', SESSION_ENHANCEMENT, '\tMINIMUM_SESSION_LENGTH: ', MINIMUM_SESSION_LENGTH)
    with Pool(works) as p:
        user_group = data.groupby("user_id")
        session_list = []
        for res in tqdm(p.imap_unordered(get_session, user_group),
                        total=len(user_group)):
            if res:
                session_list.extend(res)
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
        train_list, test_list,category, session_info = exclude_item_notin_training(train_list, test_list, category, session_info)
    print('session_info: ', session_info)
    save_data(save_path=save_path, session_info=session_info, category=category, train_list=train_list, test_list=test_list, item_category=item_category)
    
    return