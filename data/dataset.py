import os
import json
import torch
import pickle
import warnings
import numpy as np
from torch.utils import data
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore", category=UserWarning, message=".*Creating a tensor from a list of numpy.ndarrays is extremely slow.*")

def load_data(dataset_dir, validation_flag=True, valid_portion=0.2):
    '''
    Args:
        dataset_dir
        valid_portion
    Returns:
        train, valid, test
        item_category
    '''
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

    if validation_flag:
        train_set, valid_set = train_test_split(train_set, test_size=valid_portion, random_state=123)
    else:
        valid_set = test_set
    item_category.insert(0, 0)
    item_category = torch.tensor(item_category).long()
    return train_set, valid_set, test_set, item_category, session_info


def process_adj_matrix(solid_adj_matrix):
    return solid_adj_matrix


def sess_collate_fn(batch):
    sess_categories_batch = []
    sess_target_batch = []
    sess_nodes_batch = []
    adj_matrix_batch = []
    nodes_categories_matrix_batch = [] 
    mask_item_batch = []
    mask_catgy_batch = []
    session_last_item_index_batch = []
    session_last_catgy_index_batch = []
    num_nodes = []
    num_sess_batch = []

    for sess in batch:
        sess_items = sess[0][0]
        num_sess_batch.append(len(sess_items))
        num_nodes.append(len(np.unique(sess_items)))
    max_session_len = np.max(num_sess_batch)
    max_nodes_len = np.max(num_nodes)

    for session, target in batch:
        sess_target_batch.append(target)
        sess_catgories = session[1]
        sess_categories_batch.append(sess_catgories + ((max_session_len - len(sess_catgories)) * [0]))
        session_last_catgy_index_batch.append(len(sess_catgories) - 1)
        sess_items = session[0]
        last_item = sess_items[-1]
        u, ind = np.unique(sess_items, return_index=True)
        unique_items = u[np.argsort(ind)]
        sess_nodes_batch.append(unique_items.tolist() + ((max_nodes_len - len(unique_items)) * [0]))
        session_last_item_index_batch.append((np.where(unique_items == last_item))[0][0])

        solid_adj_matrix = np.zeros((max_nodes_len, max_nodes_len))
        nodes_categories_matrix = np.zeros((max_nodes_len, max_session_len))
        mask = np.zeros(max_nodes_len)
        mask_catgy = np.zeros(max_session_len)
        sess_len = len(sess_items)
        mask_catgy[sess_len - 1] = 1
        for i in np.arange(sess_len - 1):
            mask_catgy[i] = 1
            u = np.where(unique_items == sess_items[i])[0][0]  
            if not mask[u]:
                mask[u] = 1
            v = np.where(unique_items == sess_items[i + 1])[0][0]
            if not solid_adj_matrix[u][v]:
                solid_adj_matrix[u][v] = 1
            if not nodes_categories_matrix[u][i]:
                nodes_categories_matrix[u][i] = 1
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
        sess_target_batch).long(), torch.tensor(session_last_item_index_batch).long(), torch.tensor(
        mask_item_batch).long(),torch.tensor(session_last_catgy_index_batch).long(), torch.tensor(mask_catgy_batch).long()

class RsData(data.Dataset):
    def __init__(self, log_dir_txt, data, sliding_size=2):
        self.data = data
        self.sliding_size = sliding_size

    def __getitem__(self, index):
        user_id, start_time, end_time, item_list, category_list, target_item, target_category = self.data[index]
        return (item_list, category_list), (target_item, target_category)

    def __len__(self):
        return len(self.data)

    def get_adj_matrix(self, sess_items, sliding_size=2):
        '''
        Args:
            sess_items: an item sequence
        Returns:
            adj_matrix: (solid_adj_matrix, dashed_adj_matrix), dtype = np.array, the adjacency matrix of a session
        '''
        unique_items = np.unique(sess_items)
        m_len = len(unique_items)
        solid_adj_matrix = np.zeros((m_len, m_len))
        dashed_adj_matrix = np.zeros((m_len, m_len))
        for i in np.arange(len(sess_items) - 1):
            u = np.where(unique_items == sess_items[i])[0][0]
            v = np.where(unique_items == sess_items[i + 1])[0][0]
            solid_adj_matrix[u][v] += 1
            if sliding_size < 2:
                raise Exception('sliding_size < 2, there are not dashed edges ')
            else:
                if i <= len(sess_items) - 1 - sliding_size:
                    for k in np.arange(2, sliding_size + 1):
                        v_1 = np.where(unique_items == sess_items[i + k])[0][0]
                        dashed_adj_matrix[u][v_1] += 1
                else:
                    pass
                
        return solid_adj_matrix, dashed_adj_matrix
