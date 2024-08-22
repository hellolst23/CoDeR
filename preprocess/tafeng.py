#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : tafeng.py
# Desc:
process tafeng data
"""
import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

from preprocess import *

def load_tafeng(file_path, nrows=None):
    """
    Load the tamll dataset and preprocess the data format
    Args:
        file_path: The path of raw tafeng.csv
    Returns:
        data：pandas, columns=['user_id', 'item_id', 'category_id', 'timestamp'], data['timestamp']: dtype= datetime64[ns], eg：2000-08-29
    """
    print("-----------------------Start reading csv -------------------------@ %ss" % datetime.datetime.now())
    data = pd.read_csv(file_path, nrows=nrows)
    print("-----------------------Finish reading csv-------------------------@ %ss" % datetime.datetime.now())
    data = data[['CUSTOMER_ID', 'PRODUCT_ID', 'PRODUCT_SUBCLASS', 'TRANSACTION_DT']]
    data.rename(columns={'CUSTOMER_ID': "user_id", 'PRODUCT_ID': "item_id", 'PRODUCT_SUBCLASS': "category_id",
                         'TRANSACTION_DT': "timestamp"}, inplace=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'], infer_datetime_format=True)  # dtype= datetime64[ns], eg：2000-08-29
    print('tafeng: ', data.head(10))
    return data


def tafeng_main():

    os.chdir('../')
    time_start = time.time()
    file_load = 'datasets/tafeng/ta_feng.csv'
    saving_load = 'datasets/tafeng_exclude/'  
    data = load_tafeng(file_path=file_load, nrows=None)
    beizhu = 'exculde item not in train when process test data, sess_enhancement: False'
    base_preprocessor(data, saving_load, beizhu=beizhu, sess_enhancement=False, exclude_item=True, minimun_session_length=2, minimum_occurrence=5, time_interval= 60 * 60 * 24, maximum_length=50,
                       train_size=0.8, works=16)
    print('base_preprocessor finished')
    time_end = time.time()
    print('time cost', (time_end - time_start)/60, 'min')


if __name__ == '__main__':
    #tafeng_main()
    #"""
    saving_dir = '../datasets/tafeng_exclude/'
    following_preprocessor(saving_dir)  
    #"""
