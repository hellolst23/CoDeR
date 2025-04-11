import os
import time
import torch
import pickle
import datetime
import numpy as np
from util.metric import evaluate
from util.utils import file_write

def train(log_path, model, criterion, optimizer, train_loader, candidate_category, device, epoch, opt):
    """
    Args:
        log_path: such as: output.txt
        model: proposed model
        criterion: loss function
        optimizer: optimization method
        catgy_optimizer: optimization method for catgy_task
        train_loader: train_loader
        candidate_category: torch.Tersor, n_items, the category of candidate item
        device: 'cuda:1' or 'cpu'
        epoch: int, epoch
    Returns:
        None
    """
    file_write(log_path, f'start training: {datetime.datetime.now()}')
    model.train()
    batch_loss_l2 = []
    batch_loss_click = []
    batch_loss_catgy = []
    batch_loss = []

    start_epoch = time.time()
    start_batch = time.time()
    current_batch = -1

    candidate_category = candidate_category.to(device)
    dataset_dir = os.path.join('datasets', opt.dataset)  # 'datasets/tb_sample'
    item_category_file = os.path.join(dataset_dir, 'item_category.pkl')

    with open(item_category_file, 'rb') as f:
        item_category = pickle.load(f)
    for i, (sess_nodes, sess_categories, adj_matrixes, nodes_categories, target,
            session_last_item, mask_node, session_last_catgy, mask_catgy) in enumerate(train_loader):
        '''
        dataloader
        sess_nodes_batch: torch.Tensor, dtype=torch.int64,  batch_size * max_nodes_len
        sess_categories_batch: torch.Tensor, dtype = torch.int64, batch_size * max_session_len
        adj_matrix_batch: torch.Tensor, dtype=torch.float64, batch_size *  max_nodes_len * max_nodes_len
        nodes_categories: torch.Tensor, dtype=torch.float64, batch_size * max_nodes_len* max_session_len
        sess_target_batch: torch.Tensor, dtype=torch.int64, batch_size * 2
        session_last_item:  torch.Tensor, dtype=torch.int64, batch_size
        mask_batch: torch.Tensor, dtype=torch.int64, batch_size * max_nodes_len
        '''
        target_item = target[:, 0].to(device)
        target_catgy = target[:, 1].to(device)
        sess_nodes = sess_nodes.to(device)
        sess_categories = sess_categories.to(device)
        adj_matrixes = adj_matrixes.to(device)
        nodes_categories = nodes_categories.to(device)
        session_last_item = session_last_item.to(device)
        mask_node = mask_node.to(device)
        session_last_catgy = session_last_catgy.to(device)
        mask_catgy = mask_catgy.to(device)

        optimizer.zero_grad()
        '''
        model
        nodes: torch.Tensor, dtype=torch.int64,  batch_size * max_nodes_len
        categories: torch.Tensor, dtype = torch.int64, batch_size * max_session_len
        adj: torch.Tensor, dtype=torch.float64, batch_size *  max_nodes_len* max_nodes_len
        nodes_categories: torch.Tensor, dtype=torch.float64, batch_size * max_nodes_len* max_session_len
        session_last_item: torch.Tensor, dtype=torch.int64, batch_size
        candidate_category: torch.Tensor, dtype=torch.int64, candidate item + 1
        mask_node: torch.Tensor, dtype=torch.int64, batch_size * max_nodes_len
        item_category: a dic, dtype={int: list[int]}
        '''
        result_click, catgy_click, l2_loss = model(sess_nodes, sess_categories, adj_matrixes, nodes_categories,
                                                   session_last_item, candidate_category, mask_node, item_category)
        '''
        criterion
        l2_loss: torch.Tensor, L2 regularization loss
        result_click: torch.Tensor, batch_size * n_items
        target_item: torch.Tensor, batch_size
        catgy_click: torch.Tensor, batch_size * n_categories
        target_catgy: torch.Tensor, batch_size
        '''
        loss, click_loss, catgy_task_loss = criterion(l2_loss, result_click, target_item, catgy_click, target_catgy)

        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        batch_loss.append(loss_value)
        batch_loss_click.append(click_loss.item())
        batch_loss_catgy.append(catgy_task_loss.item())
        batch_loss_l2.append(l2_loss.item())

        len_train_loader = len(train_loader)  
        if i % int(len_train_loader / 5 + 1) == 0:
            file_write(log_path,
            '[TRAIN]  [%d/%d] epoch: %d  current batch loss: %.4f (avg %.4f),current batch_loss_click: %.4f (avg '
            '%.4f),current batch_loss_catgy: %.4f (avg %.4f),  one batch: %.4f s,  train time: %.4f min'
            % (i, len_train_loader, epoch, loss_value, sum(batch_loss) / (i + 1), click_loss.item(),
                sum(batch_loss_click) / (i + 1), catgy_task_loss.item(),sum(batch_loss_catgy) / (i + 1),
                (time.time() - start_batch) / (i - current_batch), (time.time() - start_batch) / 60.0))
            current_batch = i
            start_batch = time.time()

    epoch_loss_mean = sum(batch_loss) / len_train_loader
    file_write(log_path, '\tEpoch Loss mean:\t%.3f,  finish one epoch@  %ss , training time:  %.4f min' % (
        epoch_loss_mean, datetime.datetime.now(), (time.time() - start_epoch) / 60.0))

    return

def test(opt, model, criterion, test_loader, candidate_category, device, Ks, Ks_auc=[50, 100, 200, 500]):
    """
    Args:
        opt: configurations
        model: proposed model
        criterion: loss function
        test_loader: test_loader
        candidate_category: torch.Tersor, 1 dim, len= n_items, the category of candidate item, [0, candidate_id_1, candidate_id_2, ... ]
        device: 'cuda:1' or 'cpu' or session_related
        Ks: list, dtype=int, top@k
        Ks_auc: list, dtype=int, auc@k
    Returns:
        recall_mean: list, dtype=float, recall value in the whole test data, [recall@10, recall@20, recall@40, recall@100]
        mrr_mean: list, dtype=float, mrr value in the whole test data, [mrr@10, mrr@20, mrr@40, mrr@100]
        ndcg_mean: list, dtype=float, ndcg value in the whole test data, [ndcg@10, ndcg@20, ndcg@40, ndcg@100]
        auc_mean: list, dtype=float, auc value in the whole test data, [auc@50, auc@100, auc@200, auc@500]
    """
    print('\t start predicting: ', datetime.datetime.now())
    model.eval()
    recalls, mrrs, ndcgs, aucs = [], [], [], []
    batch_catgy_click_loss, batch_click_loss, batch_rs_loss = [], [], []
    candidate_category = candidate_category.to(device)
    dataset_dir = os.path.join('datasets', opt.dataset)
    item_category_file = os.path.join(dataset_dir, 'item_category.pkl')
    with open(item_category_file, 'rb') as f:
        item_category = pickle.load(f)
    with torch.no_grad():
        for i, (sess_nodes, sess_categories, adj_matrixes, nodes_categories, target, session_last_item,
            mask_node, session_last_catgy_index, mask_catgy) in enumerate(test_loader):
            sess_nodes = sess_nodes.to(device)
            sess_categories = sess_categories.to(device)
            adj_matrixes = adj_matrixes.to(device)
            nodes_categories = nodes_categories.to(device)
            target_item = target[:, 0].to(device)
            target_catgy = target[:, 1].to(device)
            session_last_item = session_last_item.to(device)
            mask_node = mask_node.to(device)
            session_last_catgy_index = session_last_catgy_index.to(device)
            mask_catgy = mask_catgy.to(device)

            result_click, catgy_click, l2_loss = model(sess_nodes, sess_categories, adj_matrixes, nodes_categories,
                                                   session_last_item, candidate_category, mask_node, item_category)
            loss, click_loss, catgy_task_loss = criterion(l2_loss, result_click, target_item, catgy_click, target_catgy)

            batch_click_loss.append(click_loss.item())
            batch_catgy_click_loss.append(catgy_task_loss.item())
            batch_rs_loss.append(loss.item())

            result_click = result_click.detach()
            metrics = evaluate(result_click, target_item, Ks, Ks_auc)
            recalls.append(metrics['recall'])
            mrrs.append(metrics['mrr'])
            ndcgs.append(metrics['ndcg'])
            aucs.append(metrics['auc'])

    recall_mean = np.mean(recalls, axis=0) * 100
    mrr_mean = np.mean(mrrs, axis=0) * 100
    ndcg_mean = np.mean(ndcgs, axis=0) * 100
    auc_mean = np.mean(aucs, axis=0) * 100

    return recall_mean, mrr_mean, ndcg_mean, auc_mean