import os
import time
import torch
import psutil
import random
import numpy as np
from functools import wraps

import torch


def seed_torch(log_path_txt, seed=None):
    if seed is None:
        seed = int(time.time())
    file_write(log_path_txt, f'************ seed ***********: {seed}')
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@timefn: {fn.__name__} took {t2 - t1: .5f} s")
        return result
    return measure_time

def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    print("{} memory used: {} MB".format(hint, memory))


def print_result(log_file, title,metric, Ks=[10,20,50]):
    assert len(metric)==len(Ks)
    title_str = '%s@%d\t' * len(Ks)
    file_write(log_file, title_str %(title, Ks[0], title, Ks[1],title, Ks[2],title, Ks[3],title, Ks[4], title, Ks[5],
       title, Ks[6]))
    result_str = '%.4f\t' * len(Ks)
    file_write(log_file, result_str % (metric[0], metric[1], metric[2], metric[3],metric[4],
        metric[5],metric[6]))
    return


def print_auc_result(log_file,title, metric, Ks_auc=[50,100,200,500]):
    title_str = '%s@%d\t' * len(Ks_auc)
    file_write(log_file, title_str % (title, Ks_auc[0], title, Ks_auc[1], title, Ks_auc[2], title, Ks_auc[3]))
    result_str = '%.4f\t' * len(Ks_auc)
    file_write(log_file, result_str % (metric[0], metric[1], metric[2], metric[3]))

    return


def cprint(log_file, words: str):
    print(f"\033[31;1m{words}\033[0m")
    file_write(log_file, '\n', whether_print=False)
    file_write(log_file,words, whether_print=False)


def file_write(log_file, s, whether_print=True):
    load_optimal_model = False
    if whether_print:
        print(s)
    if not load_optimal_model:
        with open(log_file, 'a') as f:
            f.write(s+'\n')

def get_items_byCategory(target, item_catgy, sample_strategy=None):
    n_items = item_catgy.shape[0]
    if sample_strategy != 'category':
        return torch.arange(1,n_items).tolist()
    target_category = item_catgy[target]
    item_catgy = item_catgy.cpu().numpy()
    items_index = np.argwhere(item_catgy == target_category.cpu().numpy())  # item_num * 1
    items_index = items_index.reshape(1, -1).tolist()[0]
    return items_index

def normalize_array(arr, type):
    arr = np.array(arr, dtype=np.float32)
    min_val = np.min(arr)
    max_val = np.max(arr)
    mean_val = np.mean(arr)
    std_val = np.std(arr)

    arr = arr.astype(float)
    exp_arr = np.exp(arr)
    sum_arr = np.sum(exp_arr)

    
    if type == 'max_min':
        norm_arr = (arr - min_val) / (max_val - min_val)
    if type == 'z_score':  
        norm_arr = (arr - mean_val) / std_val
    if type == 'softmax':
        norm_arr = exp_arr / sum_arr
    if type == 'sigmoid':
        norm_arr = 1 / (1 + np.exp(-arr))
    
    return norm_arr

def normalize_array_2D(arr, type):
    min_val = np.min(arr)
    max_val = np.max(arr)
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    
    if type == 'max_min':
        norm_arr = (arr - min_val) / (max_val - min_val)
    if type == 'z_score':  
        norm_arr = (arr - mean_val) / std_val
    if type == 'softmax':
        norm_arr = np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 1, arr)
    
    return norm_arr