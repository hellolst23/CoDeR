#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : train_test.py
# Desc:
    train_test framework
"""
import time
import datetime
from tqdm import tqdm

import torch
import os
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score

import torch.nn.functional as F
from util.metric import evaluate
from util.utils import sample_negative, file_write, cprint, normalize_array
from tensorboardX import SummaryWriter
import pandas as pd
from layer.KL import get_kl_score
from layer.modularity import modularity

def train(log_path_txt, model, criterion, rec_optimizer, catgy_optimizer, train_loader, item_catgy, device, epoch, opt):
    """
        一个batch的数据跑完, 返回info loss， click loss 和 recommendation loss
        Args:
            log_path_txt: 文件路径，such as： output.txt
            model: proposed model
            criterion: loss function
            rec_optimizer: optimization method
            catgy_optimizer: optimization method for catgy_task
            train_loader: train_loader
            lamda: float, a hyper_parameter to balance the impact of info_max loss and click loss
            item_catgy: torch.Tersor, 1 dim, len= n_items, the category of candidate item, [0, candidate_id_1, candidate_id_2, ... ]
            device: 'cuda:1' or 'cpu' or session_related
            epoch: int, epoch
        Returns:
            info_loss: float, the mean of batch info_loss
            click_loss: float, the mean of batch click_loss
            rs_loss：float, the mean of batch rs_loss
            node_representation： torch.Tensor, batch_size * n_demand * max_nodes_len * embedding_dim_node, max_nodes_len: 倒数第二个batch的最大节点数
        """
    file_write(log_path_txt, f'start training: {datetime.datetime.now()}')
    model.train()
    batch_loss_l2 = []
    batch_loss_info = []
    batch_loss_click = []
    batch_loss_catgy = []
    batch_emb_sim_loss = []
    batch_loss = []  # record each batch loss for current epoch

    recalls_c, mrrs_c, ndcgs_c, aucs_c = [], [], [], []

    start_epoch = time.time()
    start_batch = time.time()
    current_batch = -1
    node_representation = None  # record the last second batch node representation

    item_catgy = item_catgy.to(device)

    # ******************************************************************************************************
    for i, (sess_nodes, sess_categories, adj_matrixes, nodes_categories_matrixes, target,
            session_last_item_index, mask_node,session_last_catgy_index, mask_catgy) in enumerate(train_loader):
        '''
        dataloader 返回值
        sess_nodes_batch： torch.Tensor, dtype=torch.int64,  batch_size * max_nodes_len
        sess_categories_batch: torch.Tensor， dtype = torch.int64, batch_size * max_session_len
        adj_matrix_batch： torch.Tensor, dtype=torch.float64, batch_size *  max_nodes_len* max_nodes_len
        nodes_categories_matrixes: torch.Tensor, dtype=torch.float64, batch_size * max_nodes_len* max_session_len
        sess_target_batch: torch.Tensor, dtype=torch.int64, batch_size * 2 , [[target_item_id, target_categories_id],...]
        session_last_item_index:  torch.Tensor, dtype=torch.int64, batch_size, record the last item index in unique nodes
        mask_batch: torch.Tensor, dtype=torch.int64, batch_size * max_nodes_len, record the clicked nodes in a session
        '''
        '''
        model forward 函数
        nodes: torch.Tensor, dtype=torch.int64,  batch_size * max_nodes_len, session unique item sequence.
        categories: torch.Tensor， dtype = torch.int64, batch_size * max_session_len, session items categories sequence.
        adj: torch.Tensor, dtype=torch.float64, batch_size *  max_nodes_len* max_nodes_len, session unique item adjacent matrix.
        nodes_categories: torch.Tensor, dtype=torch.float64, batch_size * max_nodes_len* max_session_len, session items categories to items mapping matrices.
        session_last_item_index: torch.Tensor, dtype=torch.int64, batch_size, Index of the the last item in session in sess_nodes_batch.
        candidate_category: torch.Tensor, dtype=torch.int64, candidate item + 1
        '''

        # print(f"batch_size: {adj_matrixes.shape[0]}\tmax_nodes_len: {adj_matrixes.shape[1]}\tmax_session_len: {sess_categories.shape[-1]}")
        target_item = target[:, 0].to(device)  # Note: target is a 2 dim tensor, [[item_id, category_id],...]
        target_catgy = target[:, 1].to(device)
        if opt.n_negative == 0:
            neg_sample = None
        elif opt.loss_name == 'cross':
            neg_sample = sample_negative(sess_nodes, target_item, item_catgy, n_negative=opt.n_negative,
                                         sample_strategy=opt.sample_strategy)
            neg_sample = neg_sample.to(device)
        elif opt.loss_name == 'bpr':
            neg_sample = sample_negative(sess_nodes, target_item, item_catgy, n_negative=1,
                                         sample_strategy=opt.sample_strategy)
            neg_sample = neg_sample.to(device)

        sess_nodes = sess_nodes.to(device)
        sess_categories = sess_categories.to(device)
        adj_matrixes = adj_matrixes.to(device)
        nodes_categories_matrixes = nodes_categories_matrixes.to(device)
        session_last_item_index = session_last_item_index.to(device)
        mask_node = mask_node.to(device)
        session_last_catgy_index = session_last_catgy_index.to(device)
        mask_catgy = mask_catgy.to(device)

        ############# train catgy_loss
        if opt.catgy_lamda > 0:
            catgy_optimizer.zero_grad()
            result_click, catgy_click, infomax_loss, l2_loss, gnn_node_representation, demand_sim_loss = model(sess_nodes,
                                                                                                               sess_categories,
                                                                                                               adj_matrixes,
                                                                                                               nodes_categories_matrixes,
                                                                                                               session_last_item_index,
                                                                                                               item_catgy,
                                                                                                               mask_node,session_last_catgy_index, mask_catgy)
            loss, click_loss, catgy_task_loss = criterion(infomax_loss, l2_loss, result_click, target_item, demand_sim_loss,
                                                          neg_sample, catgy_click=catgy_click, target_catgy=target_catgy)

            catgy_task_loss.backward()
            catgy_optimizer.step()
            batch_loss_catgy.append(catgy_task_loss.item())

        ############# train loss =  click_loss + self.info_lamda * info_loss + l2_loss + 0.5 * demand_sim_loss
        rec_optimizer.zero_grad()
        result_click, catgy_click, infomax_loss, l2_loss, gnn_node_representation, demand_sim_loss = model(sess_nodes,
                                                                                              sess_categories,
                                                                                              adj_matrixes,
                                                                                              nodes_categories_matrixes,
                                                                                              session_last_item_index,
                                                                                              item_catgy, mask_node, session_last_catgy_index, mask_catgy)
        """
        with SummaryWriter(comment='DemandRS') as w:
            w.add_graph(model, (sess_nodes, sess_categories, adj_matrixes, nodes_categories_matrixes,
                                session_last_item_index, item_catgy, mask_node), )

        exit()
        """
        
        # caculate the KL value
        dataset_dir = os.path.join('datasets', opt.dataset)  # 'datasets/tb_sample'
        item_category_file = os.path.join(dataset_dir, 'item_category.pkl')
     
        with open(item_category_file, 'rb') as f:
            item_category = pickle.load(f)

        category_list = np.zeros(shape=(len(item_category)))

        # CPU
        # target_item = candidate_ranking(sess_nodes.detach().numpy(), target_item.detach().numpy(), 
        #                                 item_category, category_list, adj_matrixes.detach().numpy(), sess_nodes.detach().numpy(), 
        #                                 nodes_categories_matrixes.detach().numpy(), sess_categories.detach().numpy(), 0.95, i, device) 
        
        # GPU
        target_item = candidate_ranking(sess_nodes.cpu().detach().numpy(), target_item.cpu().detach().numpy(), 
                                        item_category, category_list, adj_matrixes.cpu().detach().numpy(), sess_nodes.cpu().detach().numpy(), 
                                        nodes_categories_matrixes.cpu().detach().numpy(), sess_categories.cpu().detach().numpy(), opt.alpha, i, device) 


        loss, click_loss, catgy_task_loss = criterion(infomax_loss, l2_loss, result_click, target_item, demand_sim_loss,
                                     neg_sample, catgy_click=catgy_click, target_catgy=target_catgy)

        loss.backward()
        rec_optimizer.step()

        loss_value = loss.item()  # get python number which occupies a smaller storage
        batch_loss.append(loss_value)
        batch_loss_click.append(click_loss.item())
        if opt.catgy_lamda == 0:
            batch_loss_catgy.append(catgy_task_loss.item())
        batch_loss_info.append(infomax_loss.item())
        batch_loss_l2.append(l2_loss.item())
        batch_emb_sim_loss.append(demand_sim_loss.item())

        # category evaluation
        metrics_c = evaluate(catgy_click.max(1).values, target_catgy, Ks=[5, 10, 20, 40], Ks_auc=[100, 200])
        recalls_c.append(metrics_c['recall'])
        mrrs_c.append(metrics_c['mrr'])
        ndcgs_c.append(metrics_c['ndcg'])
        aucs_c.append(metrics_c['auc'])

        # batch_metric_result(catgy_click.max(1).values, target_catgy, log_path_txt=opt.log_path_txt)

        len_train_loader = len(train_loader)  #
        if i == len_train_loader - 2:
            node_representation = gnn_node_representation
        if i % int(len_train_loader / 5 + 1) == 0:
            file_write(log_path_txt,
                       '[TRAIN]  [%d/%d] epoch: %d  current batch loss: %.4f (avg %.4f),current batch_loss_click: %.4f (avg '
                       '%.4f),current batch_loss_catgy: %.4f (avg %.4f), current batch_loss_info: %.4f (avg %.4f), one batch: %.4f s,  train time: %.4f min'
                       % (i, len_train_loader, epoch, loss_value, sum(batch_loss) / (i + 1), click_loss.item(),
                          sum(batch_loss_click) / (i + 1), catgy_task_loss.item(),sum(batch_loss_catgy) / (i + 1) ,infomax_loss.item(), sum(batch_loss_info) / (i + 1),
                          (time.time() - start_batch) / (i - current_batch), (time.time() - start_batch) / 60.0))
            if opt.catgy_lamda > 0:
                cprint(log_path_txt, f'Current catgy prediction result: 5, \t10, \t20 , \t40')
                file_write(log_path_txt, f"Recall {np.mean(recalls_c, axis=0) * 100}")
                file_write(log_path_txt, f"MRR'{np.mean(mrrs_c, axis=0) * 100}")
                file_write(log_path_txt, f"NDCG', {np.mean(ndcgs_c, axis=0) * 100}")
                file_write(log_path_txt, f"AUC', {np.mean(aucs_c, axis=0) * 100}")

            current_batch = i
            start_batch = time.time()

    epoch_loss_mean = sum(batch_loss) / len_train_loader
    epoch_loss_click = sum(batch_loss_click) / len_train_loader
    epoch_loss_info = sum(batch_loss_info) / len_train_loader
    file_write(log_path_txt, '\tEpoch Loss mean:\t%.3f,  finish one epoch@  %ss , training time:  %.4f min' % (
        epoch_loss_mean, datetime.datetime.now(), (time.time() - start_epoch) / 60.0))

    return epoch_loss_mean, epoch_loss_click, epoch_loss_info, node_representation


# def train(log_path_txt, model, criterion, rec_optimizer, catgy_optimizer, train_loader, item_catgy, device, epoch, opt):
#     """
#         一个batch的数据跑完, 返回info loss， click loss 和 recommendation loss
#         Args:
#             log_path_txt: 文件路径，such as： output.txt
#             model: proposed model
#             criterion: loss function
#             rec_optimizer: optimization method
#             catgy_optimizer: optimization method for catgy_task
#             train_loader: train_loader
#             lamda: float, a hyper_parameter to balance the impact of info_max loss and click loss
#             item_catgy: torch.Tersor, 1 dim, len= n_items, the category of candidate item, [0, candidate_id_1, candidate_id_2, ... ]
#             device: 'cuda:1' or 'cpu' or session_related
#             epoch: int, epoch
#         Returns:
#             info_loss: float, the mean of batch info_loss
#             click_loss: float, the mean of batch click_loss
#             rs_loss：float, the mean of batch rs_loss
#             node_representation： torch.Tensor, batch_size * n_demand * max_nodes_len * embedding_dim_node, max_nodes_len: 倒数第二个batch的最大节点数
#         """
#     file_write(log_path_txt, f'start training: {datetime.datetime.now()}')
#     model.train()
#     batch_loss_l2 = []
#     batch_loss_info = []
#     batch_loss_click = []
#     batch_loss_catgy = []
#     batch_emb_sim_loss = []
#     batch_loss = []  # record each batch loss for current epoch

#     recalls_c, mrrs_c, ndcgs_c, aucs_c = [], [], [], []

#     start_epoch = time.time()
#     start_batch = time.time()
#     current_batch = -1
#     node_representation = None  # record the last second batch node representation

#     item_catgy = item_catgy.to(device)

#     # ******************************************************************************************************
#     for i, (sess_nodes, sess_categories, adj_matrixes, nodes_categories_matrixes, target,
#             session_last_item_index, mask_node,session_last_catgy_index, mask_catgy) in enumerate(train_loader):
#         '''
#         dataloader 返回值
#         sess_nodes_batch： torch.Tensor, dtype=torch.int64,  batch_size * max_nodes_len
#         sess_categories_batch: torch.Tensor， dtype = torch.int64, batch_size * max_session_len
#         adj_matrix_batch： torch.Tensor, dtype=torch.float64, batch_size *  max_nodes_len* max_nodes_len
#         nodes_categories_matrixes: torch.Tensor, dtype=torch.float64, batch_size * max_nodes_len* max_session_len
#         sess_target_batch: torch.Tensor, dtype=torch.int64, batch_size * 2 , [[target_item_id, target_categories_id],...]
#         session_last_item_index:  torch.Tensor, dtype=torch.int64, batch_size, record the last item index in unique nodes
#         mask_batch: torch.Tensor, dtype=torch.int64, batch_size * max_nodes_len, record the clicked nodes in a session
#         '''
#         '''
#         model forward 函数
#         nodes: torch.Tensor, dtype=torch.int64,  batch_size * max_nodes_len, session unique item sequence.
#         categories: torch.Tensor， dtype = torch.int64, batch_size * max_session_len, session items categories sequence.
#         adj: torch.Tensor, dtype=torch.float64, batch_size *  max_nodes_len* max_nodes_len, session unique item adjacent matrix.
#         nodes_categories: torch.Tensor, dtype=torch.float64, batch_size * max_nodes_len* max_session_len, session items categories to items mapping matrices.
#         session_last_item_index: torch.Tensor, dtype=torch.int64, batch_size, Index of the the last item in session in sess_nodes_batch.
#         candidate_category: torch.Tensor, dtype=torch.int64, candidate item + 1
#         '''

#         # print(f"batch_size: {adj_matrixes.shape[0]}\tmax_nodes_len: {adj_matrixes.shape[1]}\tmax_session_len: {sess_categories.shape[-1]}")
#         target_item = target[:, 0].to(device)  # Note: target is a 2 dim tensor, [[item_id, category_id],...]
#         target_catgy = target[:, 1].to(device)
#         if opt.n_negative == 0:
#             neg_sample = None
#         elif opt.loss_name == 'cross':
#             neg_sample = sample_negative(sess_nodes, target_item, item_catgy, n_negative=opt.n_negative,
#                                          sample_strategy=opt.sample_strategy)
#             neg_sample = neg_sample.to(device)
#         elif opt.loss_name == 'bpr':
#             neg_sample = sample_negative(sess_nodes, target_item, item_catgy, n_negative=1,
#                                          sample_strategy=opt.sample_strategy)
#             neg_sample = neg_sample.to(device)

#         sess_nodes = sess_nodes.to(device)
#         sess_categories = sess_categories.to(device)
#         adj_matrixes = adj_matrixes.to(device)
#         nodes_categories_matrixes = nodes_categories_matrixes.to(device)
#         session_last_item_index = session_last_item_index.to(device)
#         mask_node = mask_node.to(device)
#         session_last_catgy_index = session_last_catgy_index.to(device)
#         mask_catgy = mask_catgy.to(device)

#         ############# train catgy_loss
#         if opt.catgy_lamda > 0:
#             catgy_optimizer.zero_grad()
#             result_click, catgy_click, infomax_loss, l2_loss, gnn_node_representation, demand_sim_loss = model(sess_nodes,
#                                                                                                                sess_categories,
#                                                                                                                adj_matrixes,
#                                                                                                                nodes_categories_matrixes,
#                                                                                                                session_last_item_index,
#                                                                                                                item_catgy,
#                                                                                                                mask_node,session_last_catgy_index, mask_catgy)
#             loss, click_loss, catgy_task_loss = criterion(infomax_loss, l2_loss, result_click, target_item, demand_sim_loss,
#                                                           neg_sample, catgy_click=catgy_click, target_catgy=target_catgy)

#             catgy_task_loss.backward()
#             catgy_optimizer.step()
#             batch_loss_catgy.append(catgy_task_loss.item())

#         ############# train loss =  click_loss + self.info_lamda * info_loss + l2_loss + 0.5 * demand_sim_loss
#         rec_optimizer.zero_grad()
#         result_click, catgy_click, infomax_loss, l2_loss, gnn_node_representation, demand_sim_loss = model(sess_nodes,
#                                                                                               sess_categories,
#                                                                                               adj_matrixes,
#                                                                                               nodes_categories_matrixes,
#                                                                                               session_last_item_index,
#                                                                                               item_catgy, mask_node, session_last_catgy_index, mask_catgy)
#         """
#         with SummaryWriter(comment='DemandRS') as w:
#             w.add_graph(model, (sess_nodes, sess_categories, adj_matrixes, nodes_categories_matrixes,
#                                 session_last_item_index, item_catgy, mask_node), )

#         exit()
#         """
        
#         # caculate the KL value
#         # dataset_dir = os.path.join('datasets', opt.dataset)  # 'datasets/tb_sample'
#         # item_category_file = os.path.join(dataset_dir, 'item_category.pkl')
     
#         # with open(item_category_file, 'rb') as f:
#         #     item_category = pickle.load(f)

#         # category_list = np.zeros(shape=(len(item_category)))

#         # final_score = candidate_ranking(sess_nodes.detach().numpy(), target_item.detach().numpy(), 
#         #                                 item_category, category_list, result_click.detach().numpy(), 0.01) # alpha = 0.01
#         # print('Final_score: ')
#         # print(final_score)

#         # caculate the modularity
#         # node_modularity, categories_modularity = get_modularity(adj_matrixes.detach().numpy(), sess_nodes.detach().numpy(), 
#         #                                                         nodes_categories_matrixes.detach().numpy(), sess_categories.detach().numpy())
#         # print('node_modularity: ')
#         # print(node_modularity)
#         # print('categories_modularity: ')
#         # print(categories_modularity)


#         # if i < 79:
#         # # i = min(i, 99)

#         #     P_dr = get_confounder(adj_matrixes[i].detach().numpy(), sess_nodes[i].detach().numpy(), target_item.detach().numpy(), 
#         #                         sess_categories[i].detach().numpy(), confounder_prior=None)
#         #     # print('P_dr: ')
#         #     # print(P_dr)
#         #     target_item_confounder = P_dr * target_item.numpy()
#         #     target_item = torch.from_numpy(target_item_confounder).long()
#         #     target_item = target_item.to(device)
        

#         loss, click_loss, catgy_task_loss = criterion(infomax_loss, l2_loss, result_click, target_item, demand_sim_loss,
#                                      neg_sample, catgy_click=catgy_click, target_catgy=target_catgy)

#         loss.backward()
#         rec_optimizer.step()

#         loss_value = loss.item()  # get python number which occupies a smaller storage
#         batch_loss.append(loss_value)
#         batch_loss_click.append(click_loss.item())
#         if opt.catgy_lamda == 0:
#             batch_loss_catgy.append(catgy_task_loss.item())
#         batch_loss_info.append(infomax_loss.item())
#         batch_loss_l2.append(l2_loss.item())
#         batch_emb_sim_loss.append(demand_sim_loss.item())

#         # category evaluation
#         metrics_c = evaluate(catgy_click.max(1).values, target_catgy, Ks=[5, 10, 20, 40], Ks_auc=[100, 200])
#         recalls_c.append(metrics_c['recall'])
#         mrrs_c.append(metrics_c['mrr'])
#         ndcgs_c.append(metrics_c['ndcg'])
#         aucs_c.append(metrics_c['auc'])

#         # batch_metric_result(catgy_click.max(1).values, target_catgy, log_path_txt=opt.log_path_txt)

#         len_train_loader = len(train_loader)  #
#         if i == len_train_loader - 2:
#             node_representation = gnn_node_representation
#         if i % int(len_train_loader / 5 + 1) == 0:
#             file_write(log_path_txt,
#                        '[TRAIN]  [%d/%d] epoch: %d  current batch loss: %.4f (avg %.4f),current batch_loss_click: %.4f (avg '
#                        '%.4f),current batch_loss_catgy: %.4f (avg %.4f), current batch_loss_info: %.4f (avg %.4f), one batch: %.4f s,  train time: %.4f min'
#                        % (i, len_train_loader, epoch, loss_value, sum(batch_loss) / (i + 1), click_loss.item(),
#                           sum(batch_loss_click) / (i + 1), catgy_task_loss.item(),sum(batch_loss_catgy) / (i + 1) ,infomax_loss.item(), sum(batch_loss_info) / (i + 1),
#                           (time.time() - start_batch) / (i - current_batch), (time.time() - start_batch) / 60.0))
#             if opt.catgy_lamda > 0:
#                 cprint(log_path_txt, f'Current catgy prediction result: 5, \t10, \t20 , \t40')
#                 file_write(log_path_txt, f"Recall {np.mean(recalls_c, axis=0) * 100}")
#                 file_write(log_path_txt, f"MRR'{np.mean(mrrs_c, axis=0) * 100}")
#                 file_write(log_path_txt, f"NDCG', {np.mean(ndcgs_c, axis=0) * 100}")
#                 file_write(log_path_txt, f"AUC', {np.mean(aucs_c, axis=0) * 100}")

#             current_batch = i
#             start_batch = time.time()

#     epoch_loss_mean = sum(batch_loss) / len_train_loader
#     epoch_loss_click = sum(batch_loss_click) / len_train_loader
#     epoch_loss_info = sum(batch_loss_info) / len_train_loader
#     file_write(log_path_txt, '\tEpoch Loss mean:\t%.3f,  finish one epoch@  %ss , training time:  %.4f min' % (
#         epoch_loss_mean, datetime.datetime.now(), (time.time() - start_epoch) / 60.0))

#     return epoch_loss_mean, epoch_loss_click, epoch_loss_info, node_representation

# def batch_metric_result(click, target, log_path_txt, Ks=[5, 10, 20 , 40], Ks_auc=[100, 200]):
#     """

#     :param click:
#     :param target:
#     :return:
#     """
#     metrics = evaluate(click, target, Ks, Ks_auc)
#     cprint(log_path_txt, f'Current batch: 5, \t10, \t20 , \t40')
#     file_write(log_path_txt, f"Recall { metrics['recall']}")
#     file_write(log_path_txt, f"MRR'{metrics['mrr']}")
#     file_write(log_path_txt, f"NDCG', {metrics['ndcg']}")
#     file_write(log_path_txt, f"AUC', {metrics['auc']}")


def test(opt,epoch, model, criterion, test_loader, item_catgy, device, Ks, Ks_auc=[50, 100, 200, 500],
         visualize=False):
    """
    test the model for one epoch, return recall and mrr
    Args:
        model: proposed model
        criterion: loss function
        test_loader: test_loader
        item_catgy: torch.Tersor, 1 dim, len= n_items, the category of candidate item, [0, candidate_id_1, candidate_id_2, ... ]
        device: 'cuda:1' or 'cpu' or session_related
        Ks: list, dtype=int, top@k for k in Ks, such as [10.20.50,100]
        Ks_auc: list, dtype=int, auc@k for k in Ks_auc, such as [50,100, 200,500], where k is the number of negative samples
        visualize: case study for demands
    Returns:
        recall_mean: list, dtype=float, recall value in the whole test data, [recall@10, recall@20, recall@50, recall@100]
        mrr_mean:  list, dtype=float, mrr value in the whole test data, [mrr@10, mrr@20, mrr@50, mrr@100]
        info_loss_mean: float, info_loss mean of the whole test data
        click_loss_mean: float, click prediction loss of the whole test data
        rs_loss_mean: float, rs_loss = info_loss + click_loss
    """
    log_path_txt = opt.log_path_txt
    print('\t start predicting: ', datetime.datetime.now())
    model.eval()
    recalls, mrrs, ndcgs, aucs = [], [], [], []
    recalls_c, mrrs_c, ndcgs_c, aucs_c = [], [], [], []
    batch_info_loss, batch_catgy_click_loss, batch_click_loss, batch_rs_loss = [], [], [], []
    item_catgy = item_catgy.to(device)
    if visualize:
        visualized_result = []  # item_ids, categories_ids, demand_score(demandA|demandB), Top5 predict_ID, Top 5 PVSD(demandA|demandB)
    with torch.no_grad():
        for i, (sess_nodes, sess_categories, adj_matrixes, nodes_categories_matrixes, target, session_last_item_index,
             mask_node, session_last_catgy_index, mask_catgy) in enumerate(test_loader):
            sess_nodes = sess_nodes.to(device)
            sess_categories = sess_categories.to(device)
            adj_matrixes = adj_matrixes.to(device)
            nodes_categories_matrixes = nodes_categories_matrixes.to(device)
            target_item = target[:, 0].to(device)
            target_catgy = target[:, 1].to(device)
            session_last_item_index = session_last_item_index.to(device)
            mask_node = mask_node.to(device)
            session_last_catgy_index = session_last_catgy_index.to(device)
            mask_catgy = mask_catgy.to(device)

            result_click, catgy_click, infomax_loss, l2_loss, _, demand_sim_loss = model(sess_nodes, sess_categories, adj_matrixes,
                                                                                            nodes_categories_matrixes,
                                                                                            session_last_item_index, item_catgy,
                                                                                            mask_node, session_last_catgy_index, mask_catgy)
            if visualize:
                demand_score, p_v_s_d = model.get_demand_Score()
                int2str = lambda l: [str(x) for x in l]
                float2str = lambda l: ["%.2f" % x for x in l]
                for i in range(sess_nodes.shape[0]):
                    item_ids = sess_nodes[i].detach().cpu().numpy().tolist()  # Max length
                    item_ids = "|".join(int2str(item_ids))
                    categories_ids = sess_categories[i].detach().cpu().numpy().tolist()  # Max length
                    categories_ids = "|".join(int2str(categories_ids))
                    d_score = demand_score[i]  # Max length * n_demand * 1
                    demand_vis = []
                    for id in range(d_score.shape[1]):
                        s = []
                        for d in range(d_score.shape[0]):
                            s.append(d_score[d, id].item())
                        demand_vis.append("(" + ",".join(float2str(s)) + ")")
                    demand_vis = "|".join(demand_vis)
                    p_d_score = p_v_s_d[i]  # n_demand * candidate_score
                    topk_id = torch.topk(result_click[i], k=5, sorted=True).indices
                    topk_p_d_score = p_d_score[:, topk_id]
                    top_k_p = []
                    for id in range(topk_p_d_score.shape[1]):
                        s = []
                        for d in range(d_score.shape[0]):
                            s.append(topk_p_d_score[d, id].item())
                        top_k_p.append("(" + ",".join(float2str(s)) + ")")
                    top_k_p = "|".join(top_k_p)
                    topk_id = "|".join(int2str(topk_id.cpu().numpy().tolist()))
                    visualized_result.append((item_ids, categories_ids, demand_vis, topk_id, top_k_p))
                vis = pd.DataFrame(visualized_result,
                                   columns=["item_ids", "categories_ids", "demand_score", "Top5 predict_ID",
                                            "Top 5 PVSD"])
                vis.to_csv("result_tmall.csv")
            # if epoch == 8:
            #     print(sess_nodes)
            #     print(sess_categories)
            #     print(target)
            #     print(torch.topk(result_click, k=20, dim=-1)[1])
            dataset_dir = os.path.join('datasets', opt.dataset)  # 'datasets/tb_sample'
            item_category_file = os.path.join(dataset_dir, 'item_category.pkl')
     
            with open(item_category_file, 'rb') as f:
                item_category = pickle.load(f)

            category_list = np.zeros(shape=(len(item_category)))

            # CPU
            # target_item = candidate_ranking(sess_nodes.detach().numpy(), target_item.detach().numpy(), 
            #                             item_category, category_list, adj_matrixes.detach().numpy(), sess_nodes.detach().numpy(), 
            #                             nodes_categories_matrixes.detach().numpy(), sess_categories.detach().numpy(), 0.95, i, device) 

            # GPU
            target_item = candidate_ranking(sess_nodes.cpu().detach().numpy(), target_item.cpu().detach().numpy(), 
                                        item_category, category_list, adj_matrixes.cpu().detach().numpy(), sess_nodes.cpu().detach().numpy(), 
                                        nodes_categories_matrixes.cpu().detach().numpy(), sess_categories.cpu().detach().numpy(), opt.alpha, i, device) 
            # target_item = target_item.to(device)


            loss, click_loss, catgy_task_loss = criterion(infomax_loss, l2_loss, result_click, target_item, demand_sim_loss, neg_sample=None, catgy_click=catgy_click, target_catgy=target_catgy)
            batch_info_loss.append(infomax_loss.item())
            batch_click_loss.append(click_loss.item())
            batch_catgy_click_loss.append(catgy_task_loss.item())
            batch_rs_loss.append(loss.item())

            # result_click = F.softmax(result_click, dim=1).detach().to('cpu')
            result_click = result_click.detach()

            # item evaluation
            metrics = evaluate(result_click, target_item, Ks, Ks_auc)
            recalls.append(metrics['recall'])
            mrrs.append(metrics['mrr'])
            ndcgs.append(metrics['ndcg'])
            aucs.append(metrics['auc'])

            # category evaluation
            metrics_c = evaluate(catgy_click.max(1).values, target_catgy, Ks=[5, 10, 20, 40], Ks_auc=[100, 200])
            recalls_c.append(metrics_c['recall'])
            mrrs_c.append(metrics_c['mrr'])
            ndcgs_c.append(metrics_c['ndcg'])
            aucs_c.append(metrics_c['auc'])

    # item
    recall_mean = np.mean(recalls, axis=0) * 100
    mrr_mean = np.mean(mrrs, axis=0) * 100
    ndcg_mean = np.mean(ndcgs, axis=0) * 100
    auc_mean = np.mean(aucs, axis=0) * 100

    info_loss_mean = np.mean(batch_info_loss)
    click_loss_mean = np.mean(batch_click_loss)
    rs_loss_mean = np.mean(batch_rs_loss)

    # category
    if opt.catgy_lamda > 0:
        cprint(log_path_txt, f'Current catgy prediction result: 5, \t10, \t20 , \t40')
        file_write(log_path_txt, f"Recall {np.mean(recalls_c, axis=0) * 100}")
        file_write(log_path_txt, f"MRR'{np.mean(mrrs_c, axis=0) * 100}")
        file_write(log_path_txt, f"NDCG', {np.mean(ndcgs_c, axis=0) * 100}")
        file_write(log_path_txt, f"AUC', {np.mean(aucs_c, axis=0) * 100}")
    return recall_mean, mrr_mean, ndcg_mean, auc_mean, info_loss_mean, click_loss_mean, rs_loss_mean

def candidate_ranking(training_set_index, validation_set_index, item_category, category_list, 
                      adj_matrixes, session_nodes, nodes_categories_matrixes, sess_categories, alpha, i, device):
    """
        evaluate the performance of top-n ranking 
    """
    user_kl_score, avarage_kl_score, max_kl_score, min_kl_score = get_kl_score(training_set_index, validation_set_index, 
                                                                                item_category, category_list)
    target_item = validation_set_index
    # caculate the modularity before 
    node_modularity,__ = get_modularity(adj_matrixes, session_nodes, nodes_categories_matrixes, sess_categories)

    # caculate the modularity after
    now_modularity = [0 for i in range(len(target_item))] # number(adj_matrixes)/user * target_item
    if i <=69:
        add_location = len(session_nodes[i])-1
        for j in range(len(session_nodes[i])):
            if session_nodes[i][j] == 0:
                add_location = j
                break
    else:
        add_location = len(session_nodes[0])-1
        i = 0

    for target_len, add_item in enumerate(target_item):
        temp_session_nodes = session_nodes[i]
        temp_sess_categories = np.nan_to_num(sess_categories[i])
        temp_session_nodes[add_location] = add_item
        temp_sess_categories[add_location] = item_category[add_item][0]
        solid_adj_matrix, _ = get_adj_matrix(temp_session_nodes, sliding_size=2)

        now_modularity[target_len] = modularity(solid_adj_matrix, temp_sess_categories)

    if user_kl_score[i] < avarage_kl_score:
        if abs(node_modularity[i] - np.mean(now_modularity)) < alpha:
            P_dr = get_confounder(adj_matrixes[i], session_nodes[i], target_item, sess_categories[i], confounder_prior=None)

            target_item_confounder = P_dr * target_item
            target_item = torch.from_numpy(target_item_confounder).long()
            target_item = target_item.to(device)

        else:
            target_item = torch.from_numpy(target_item).long()
            target_item = target_item.to(device)
    else:
        target_item = torch.from_numpy(target_item).long()
        target_item = target_item.to(device)

    return target_item

def get_modularity(adj_matrixes, session_nodes, nodes_categories_matrixes, sess_categories):
    """
        get the modularity of the demand graph
    """
    node_modularity = np.zeros(len(adj_matrixes), dtype=float)
    categories_modularity = np.zeros(len(nodes_categories_matrixes), dtype=float)
    # adj_matrixes = np.where(adj_matrixes == 0, 0.000001, adj_matrixes)
    # session_nodes = np.where(session_nodes == 0, 0.000001, session_nodes)
    # nodes_categories_matrixes = np.where(nodes_categories_matrixes == 0, 0.000001, nodes_categories_matrixes)
    # sess_categories = np.where(sess_categories == 0, 0.000001, sess_categories)
    np.seterr(divide='ignore',invalid='ignore')
    for userID in range(len(adj_matrixes)):
        node_modularity[userID] = modularity(np.nan_to_num(adj_matrixes[userID]), np.nan_to_num(session_nodes[userID]))
        categories_modularity[userID] = modularity(np.nan_to_num(nodes_categories_matrixes[userID]), np.nan_to_num(sess_categories[userID]))
    
    node_modularity = np.nan_to_num(node_modularity)
    avarage_node_modularity = np.mean(node_modularity)
    categories_modularity = np.nan_to_num(categories_modularity)
    avarage_categories_modularity = np.mean(categories_modularity)
    
    return node_modularity,  categories_modularity

def get_confounder_all(adj_matrixes, session_nodes, target_item, sess_categories, confounder_prior):
    """
         Args:
             session_nodes: [[item_id_0, ... ],...], 多个用户的session序列
             sess_categories: [[catid_0, ... ],...], 多个用户的类别session序列
    """
    demand_categories = len(sess_categories)

    # P(u)
    number_categories = max(max(x) for x in sess_categories)
    
    P_u = np.zeros(number_categories, dtype=float)

    if confounder_prior is not None:
        for session in sess_categories:
            for cat in session:
                P_u[cat] = P_u[cat]+1
        P_u = P_u/number_categories

    else:
        P_u = torch.tensor([1.0/number_categories for x in range(number_categories)]).unsqueeze(dim=-1)
        # p_u = torch.tensor(confounder_prior, dtype=torch.float32).unsqueeze(dim=-1)

    # P(Ddr) the modularity before adding candidate items
    # 
    # P_dr = np.zeros(len(adj_matrixes), dtype=float)
    # np.seterr(divide='ignore',invalid='ignore')
    # for userID in range(len(adj_matrixes)):
    #     P_dr[userID] = modularity(np.nan_to_num(adj_matrixes[userID]), np.nan_to_num(session_nodes[userID]))
    
    # P_dr = np.nan_to_num(P_dr)
    # P_dr = np.abs(P_dr)
    # P_dr = P_dr / len(adj_matrixes)

    # P(Ddr) the modularity of adding candidate items
    # P_dr = np.zeros(len(target_item), dtype=float)
    
    P_dr = [[0 for j in range(len(target_item))] for i in range(len(adj_matrixes))] # number(adj_matrixes)/user * target_item

    np.seterr(divide='ignore',invalid='ignore')

    add_location = np.zeros(len(adj_matrixes), dtype=int)
    for userID in range(len(adj_matrixes)):
        for i in range(len(session_nodes[userID])):
            if session_nodes[userID][i] == 0:
                add_location[userID] = i
                break

    for userID in range(len(adj_matrixes)):
        for target_len, add_item in enumerate(target_item):
            temp_session_nodes = session_nodes[userID]
            temp_session_nodes[add_location[userID]] = add_item
            solid_adj_matrix, _ = get_adj_matrix(temp_session_nodes, sliding_size=2)

            P_dr[userID][target_len] = modularity((solid_adj_matrix), np.nan_to_num(sess_categories[userID]))
    
    P_dr = np.nan_to_num(P_dr)
    P_dr = np.abs(P_dr)
    P_dr = normalize_array(P_dr, type ='softmax') # max_min/z_score/softmax
    return P_u, P_dr

def get_confounder(adj_matrixes, session_nodes, target_item, sess_categories, confounder_prior):
    """
        get the confounder of the demand graph
         Args:
             session_nodes: [item_id_0, ... ], 一个用户的session序列
             sess_categories: [catid_0, ... ],一个用户的类别session序列

    """
    # P(Ddr) the modularity before adding candidate items

    P_dr = [0 for i in range(len(target_item))] # number(adj_matrixes)/user * target_item

    np.seterr(divide='ignore',invalid='ignore')
    
    add_location = len(session_nodes)-1

    for i in range(len(session_nodes)):
        if session_nodes[i] == 0:
            add_location = i
            break


    for target_len, add_item in enumerate(target_item):
        temp_session_nodes = session_nodes
        temp_session_nodes[add_location] = add_item
        solid_adj_matrix, _ = get_adj_matrix(temp_session_nodes, sliding_size=2)

        P_dr[target_len] = modularity((solid_adj_matrix), np.nan_to_num(sess_categories))
    
    P_dr = np.nan_to_num(P_dr)
    P_dr = np.abs(P_dr)
    P_dr = normalize_array(P_dr, type ='softmax') # max_min/z_score/softmax
    return P_dr

def get_adj_matrix(sess_items, sliding_size=2):
        '''
        获取一个session的邻接矩阵
        Args:
            sess_items: [item_id_0, ... ], 一个session序列

        Returns:
            adj_matrix：(solid_adj_matrix, dashed_adj_matrix), dtype = np.array
            the adjacency matrix of a session, 索引按照该session的item_id 从小到大的顺序排列, 实线连接(有向，只算了入度的)，和虚线连接的分别用一个矩阵存起来
            example： sess_items = [6,3,4,3,1,5]
            solid_adj_matrix = [[0,0,0,0,1],
                                [1,0,1,0,0]，
                                [0,1,0,0,0],
                                [0,0,0,0,0],
                                [0,1,0,0,0]]
            dashed_adj_matrix = [[0,0,0,0,0], #sliding_size = 3
                                [1,1,0,1,0]，
                                [1,0,0,1,0],
                                [0,0,0,0,0],
                                [0,1,1,0,0]]
        '''
        unique_items = np.unique(sess_items)
        m_len = len(unique_items)
        solid_adj_matrix = np.zeros((m_len, m_len))
        dashed_adj_matrix = np.zeros((m_len, m_len))
        for i in np.arange(len(sess_items) - 1):  
            u = np.where(unique_items == sess_items[i])[0][0]  # np.where() 返回一个元组， 元组元素只有一个，为一个array数组， 记录node array数组中值为session item id 的索引
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