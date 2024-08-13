# -*- coding: utf-8 -*-
"""
# @File    : main.py

"""
#todo !!! Note: load data之后，item_catgy 首行为（0，0）

import argparse
from ast import parse
import time
import datetime
import os
import argparse
import logging

# 自动调参
import nni
from nni.utils import merge_parameter

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

from data.dataset import load_data, RsData, sess_collate_fn
from layer.loss import Loss_Diy as Loss
from layer.loss import BPRLoss
from util.train_test import train, test
from util.utils import *

# from testconfig import opt
from config import opt1 as opt_temp
if opt_temp.catgy_embedding:
    from model.DA_category import DemandAwareRS
else:
    from model.DA_dotGNN import DemandAwareRS
    #from model.DemandAwareRs import DemandAwareRS

logger = logging.getLogger(" DemandRS")


def get_params():
    """
    contain parameters by nni set

    """
    parser = argparse.ArgumentParser(description='parameters by nni set')
    parser.add_argument("--dataset", type=str, default="tafeng_0117")#tmall_minoccur30_0321")
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    # model 超参
    parser.add_argument('--hidden_size', type=int, default=160, help='hidden state size ')
    parser.add_argument('--catgy_lamda', type=float, default=0.001, help='hyper-parameter, to balance the impact between '
                                                                 'info_max loss and click loss')
    # model 结构超参1
    parser.add_argument('--n_demand', type=int, default=2, help='the number of demands in a session')
    # parser.add_argument('--dashed_order', type=int, default=2, help='the size of sliding window for construction the '
    #                                                                 'dashed adjacent matrix  ')
    # parser.add_argument('--n_gnn_layer', type=int, default=1, help='the number of gnn layer')
    #
    # model 结构超参3
    # parser.add_argument('--graph_aggregation', type=str, default='mean',
    #                     help='method to aggregate the graph, lstm, mean, '
    #                          'sum...')
    # parser.add_argument('--rs', type=str, default='dot', help='dot/mlp, p_v_s_d calculation method in RS part')
    # parser.add_argument("--nonhybrid", type=bool, default=False, help=" true：不用最后一个item， false：用最后一个item")

    # loss 超参
    parser.add_argument('--drop_catgy', type=float, default=0.25)
    parser.add_argument('--drop_item', type=float, default=0.25)
    parser.add_argument('--n_negative', type=int, default=3,
                        help='the negative samples in Loss function, if n_negative=0, it represents not carrying negative strategy')

    # parser.add_argument("--beta", type=float, default=1.0, help='hyper-parameter to balance negative samples and target sample in loss funciton')
    # parser.add_argument("--embed_l2", type=float, default=0.0001, help="l2 penalty such as [0.001, 0.0005, 0.0001, 0.00005, 0.00001]")

    # 优化超参
    # parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
    # parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    # parser.add_argument('--lr_dc_step', type=int, default=1,
    #                     help='the number of steps after which the learning rate decay')

    parser.add_argument('--seed', type=int, default=2021,
                        help='[2022, 2021,2020,2019,2018,2017,2016,2015,2014,2013]}')

    args, _ = parser.parse_known_args()
    return args

def args2config(config, args):
    """
    copy parameters from args to config
    Args:
        config: class DefaultConfig
        args: ger_params()
    Returns: class DefaultConfig

    """
    config.dataset = args['dataset']
    config.batch_size = args['batch_size']
    config.hidden_size = args['hidden_size']
    config.catgy_lamda = args['catgy_lamda']
    config.n_demand = args['n_demand']
    # config.nonhybrid = args['nonhybrid']
    config.n_negative = args['n_negative']
    config.drop_catgy = args['drop_catgy']
    config.drop_catgy = args['drop_item']
    # config.beta = args['beta']
    # config.embed_l2 = args['embed_l2']
    # config.lr = args['lr']
    # config.lr_dc = args['lr_dc']
    # config.lr_dc_step = args['lr_dc_step']
    config.seed = args['seed']

    return config

def main(args):
    opt = args2config(opt_temp, args)  # copy parameters from args to config
    opt.embedding_dim_c = opt.hidden_size
    opt.embedding_dim_i = opt.hidden_size

    time_path = time.strftime("%y%m%d-%H%M%S", time.localtime(time.time()))
    log_dir_train = os.path.join('./visual', opt.dataset, 'DemandRS',
                                 f"{time_path}_{opt.beizhu1}_{opt.n_demand}") #todo
    log_dir_train_checkpoint = os.path.join(log_dir_train, "checkpoint")
    log_path_txt = os.path.join(log_dir_train, "output.txt")
    opt.log_path_txt = log_path_txt

    if not os.path.exists(log_dir_train_checkpoint):
        os.makedirs(log_dir_train_checkpoint)

    # the head line of output file
    file_write(log_path_txt, f'{opt.beizhu}')

    opt.parse()


    # set env
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"seed----------------------:{opt.seed}")
    #  fix seed
    seed_torch(log_path_txt, opt.seed)  # default value is int(time.time())

    file_write(log_path_txt,
               f"----------------------------------- Start Loading data... @ {datetime.datetime.now()} -----------------------------------")
    dataset_dir = os.path.join('datasets', opt.dataset)  # 'datasets/tb_sample'

    train_data, valid_data, test_data, item_category, session_info = load_data(dataset_dir,
                                                                               validation_flag=opt.validation,
                                                                               valid_portion=opt.valid_portion)
    # train, valid, test: ([session_list_0, ... ],[target_0,...]), item_catgy_dict: {item_id: category_id, ...}
    file_write(log_path_txt,'-' * 45 + "Finish Loading data... @ %ss " % datetime.datetime.now() + '-' * 45)
    show_memory_info('after load_data')

    train_data = RsData(log_path_txt, train_data)
    test_data = RsData(log_path_txt, test_data)
    valid_data = RsData(log_path_txt, valid_data)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=16, shuffle=True,
                              collate_fn=sess_collate_fn)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, num_workers=16, shuffle=False,
                             collate_fn=sess_collate_fn)

    try:
        opt.n_node = session_info["num_of_item[not include 0]"] + 1
    except:
        opt.n_node = session_info["num_of_item"] + 1
    opt.n_node_c = session_info["num_of_category"] + 1
    file_write(log_path_txt, f"opt.dataset:  {opt.dataset}\topt.n_node: {opt.n_node} \topt.n_node_c: {opt.n_node_c}")

    model = DemandAwareRS(opt.n_node, opt.n_node_c, opt)
    model.to(device)

    if opt.loss_name == 'cross':
        criterion = Loss(log_path_txt, opt.n_negative, opt.sample_strategy, opt.beta)
    elif opt.loss_name == 'bpr':
        criterion = BPRLoss(log_path_txt, opt.sample_strategy)

    optimizer = optim.Adam(model.parameters(), opt.lr)
    scheduler = StepLR(optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

    Ks = [10,20,40,50,60,80,100]
    Ks_auc = [50,100,200,500]

    start = time.time()

    best_result_recall = [0]*len(Ks)
    best_result_mrr = [0] * len(Ks)
    best_result_ndcg = [0] * len(Ks)
    best_result_auc = [0] * len(Ks_auc)

    best_epoch_auc = [0] * len(Ks_auc)
    best_epoch_ndcg = [0] * len(Ks)
    best_epoch_recall = [0]*len(Ks)
    best_epoch_mrr = [0]*len(Ks)

    bad_counter = 0

    writer = SummaryWriter(log_dir=log_dir_train, comment=opt.comment)
    for epoch in range(opt.epoch):
        file_write(log_path_txt, f'epoch: {epoch}')
        # '''
        rs_loss, click_loss, info_loss, gnn_node_representation = train(log_path_txt, model, criterion, optimizer,
                                                                        train_loader, item_category, device, epoch, opt)
        # gnn_node_representation visualization
        # gnn_node_representation: batch_size * n_demand * max_nodes_len * embedding_dim_node
        last_batch_size, max_node_len = gnn_node_representation.shape[0], gnn_node_representation.shape[-2] # for 循环溢出的时最后一个batch数据，数据量不一定等于batch_size
        label_demand = []
        for i in range(opt.n_demand):
            label_demand += [i] * max_node_len
        label_demand = label_demand * last_batch_size # 区分不同的demand
        if not opt.catgy_embedding:
            gnn_node_representation = gnn_node_representation.reshape(-1, opt.hidden_size)
            writer.add_embedding(gnn_node_representation, metadata=label_demand, global_step=epoch,
                             tag='gnn_nodes_representation')
        # add loss visualization for train
        writer.add_scalars('loss', {'train_rs_loss': rs_loss}, epoch)
        writer.add_scalars('loss', {'train_click_loss': click_loss}, epoch)
        writer.add_scalars('info_loss', {'train': info_loss}, epoch)

        """
        # save current epoch model
        ckpt_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(ckpt_dict, os.path.join(log_dir_train_checkpoint,f"epoch{epoch}_checkpoint.pth.tar"))
        """
        # ************************************************ test
        # **************************************************************
        # model 加载代码

        # ckpt = torch.load(os.path.join(log_dir_train_checkpoint,f"epoch{epoch}_checkpoint.pth.tar"))
        # model.load_state_dict(ckpt['state_dict'])

        recall, mrr, ndcg, auc, info_loss, click_loss, rs_loss = test(opt,epoch, model, criterion, test_loader, opt.lamda,
                                                                      item_category, device, Ks)

        # add loss and metric visualization for test
        writer.add_scalars('info_loss', {'test': info_loss}, epoch)
        writer.add_scalars('loss', {'test_click_loss': click_loss}, epoch)
        writer.add_scalars('loss', {'test_rs_loss': rs_loss}, epoch)

        flag = 0
        for i, topk in enumerate(Ks):
            writer.add_scalars('Recall', {'Recall@{}'.format(topk): recall[i]}, epoch)
            writer.add_scalars('MRR', {'MRR@{}'.format(topk): mrr[i]}, epoch)
            writer.add_scalars('NDCG', {'NDCG@{}'.format(topk): ndcg[i]}, epoch)

            if recall[i] >= best_result_recall[i]:
                best_result_recall[i] = recall[i]
                best_epoch_recall[i] = epoch
                flag = 1
            if mrr[i] >= best_result_mrr[i]:
                best_result_mrr[i] = mrr[i]
                best_epoch_mrr[i] = epoch
                flag = 1
            if ndcg[i] >= best_result_ndcg[i]:
                best_result_ndcg[i] = ndcg[i]
                best_epoch_ndcg[i] = epoch
                flag = 1
        for i, k in enumerate(Ks_auc):
            writer.add_scalars('AUC', {'AUC@{}'.format(k): auc[i]}, epoch)
            if auc[i] >= best_result_auc[i]:
                best_result_auc[i] = auc[i]
                best_epoch_auc[i] = epoch
                flag = 1
     # print result
        cprint(log_path_txt,f'Current Result Epoch {epoch}:')
        print_result(log_path_txt, 'Recall', recall,Ks)
        print_result(log_path_txt, 'MRR', mrr,Ks)
        print_result(log_path_txt, 'NDCG', ndcg, Ks)
        print_auc_result(log_path_txt, 'AUC', auc, Ks_auc)

        file_write(log_path_txt, 'Best Result: ')
        file_write(log_path_txt, f'best_epoch_recall: {best_epoch_recall}')
        file_write(log_path_txt, f'best_epoch_mrr: {best_epoch_mrr}')
        file_write(log_path_txt, f'best_epoch_ndcg: {best_epoch_ndcg}')
        file_write(log_path_txt, f'best_epoch_auc: {best_epoch_auc}')

        print_result(log_path_txt, 'Recall', best_result_recall, Ks)
        print_result(log_path_txt, 'MRR', best_result_mrr, Ks)
        print_result(log_path_txt, 'NDCG', best_result_ndcg, Ks)
        print_auc_result(log_path_txt, 'AUC', best_result_auc, Ks_auc)

        scheduler.step(epoch=epoch)
        # early stop judgement
        bad_counter += 1 - flag  # 在各个评价指标上共出现opt.patience 次坏的结果，
        if bad_counter >= opt.patience:
            break

        # report intermediate result
        nni.report_intermediate_result(recall[2])
        logger.debug("test recall 40 %g", recall[2])
        logger.debug("Pipe send intermediate result done.")

    writer.close()
    file_write(log_path_txt, '-------------------------------------------------------')
    end = time.time()
    file_write(log_path_txt, f"Run time: {(end - start) / 60.0}min")

    # report final result
    nni.report_final_result(best_result_recall[2])
    logger.debug('Final result is %g', best_result_recall[2])
    logger.debug('Send final result done.')


if __name__ == '__main__':
    # get parameters form tuner
    print("ddddddddddd")
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(get_params(), tuner_params))
    print(params)
    main(params)


    def load_optimal_model():


        epoch = 1  # tmall:3 , tafeng:
        log_dir_train_checkpoint = "/home/ylq/git/seqRS/current/DemandRS/visual/tafeng_0117/DemandRS/210329-220729_key demand share L2正则-重要/checkpoint"
            # "/home/ylq/git/seqRS/current/DemandRS/visual/tmall_minoccur30_0321/DemandRS/211123-151351_key demand share L2正则-重要/checkpoint"

        time_path = time.strftime("%y%m%d-%H%M%S", time.localtime(time.time()))
        log_dir_train = os.path.join('./visual', opt_temp.dataset, 'DemandRS',
                                     f"加在最优模型测试_{time_path}_{opt_temp.beizhu1}")
        log_path_txt = os.path.join(log_dir_train, "output.txt")
        opt_temp.log_path_txt = log_path_txt

        #  fix seed
        seed_torch(log_path_txt, opt_temp.seed)  # default value is int(time.time())

        # set env
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt_temp.gpu_id)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset_dir = os.path.join('datasets', opt_temp.dataset)  # 'datasets/tb_sample'
        train_data, valid_data, test_data, item_category, session_info = load_data(dataset_dir,
                                                                                   validation_flag=opt_temp.validation,
                                                                                   valid_portion=opt_temp.valid_portion)
        test_data = RsData(log_path_txt, test_data)
        test_loader = DataLoader(test_data, batch_size=opt_temp.batch_size, num_workers=16, shuffle=False,
                                 collate_fn=sess_collate_fn)
        try:
            opt_temp.n_node = session_info["num_of_item[not include 0]"] + 1
        except:
            opt_temp.n_node = session_info["num_of_item"] + 1
        opt_temp.n_node_c = session_info["num_of_category"] + 1

        Ks = [10, 20, 40, 50, 60, 80, 100]
        Ks_auc = [50, 100, 200, 500]

        # model 加载代码
        ckpt = torch.load(os.path.join(log_dir_train_checkpoint,f"epoch{epoch}_checkpoint.pth.tar"))

        model = DemandAwareRS(opt_temp.n_node, opt_temp.n_node_c, opt_temp)
        model.to(device)
        if opt_temp.loss_name == 'cross':
            criterion = Loss(log_path_txt, opt_temp.n_negative, opt_temp.sample_strategy, opt_temp.beta)
        elif opt_temp.loss_name == 'bpr':
            criterion = BPRLoss(log_path_txt, opt_temp.sample_strategy)

        model.load_state_dict(ckpt['state_dict'])
        recall, mrr, ndcg, auc, info_loss, click_loss, rs_loss = test(epoch, model, criterion, test_loader, opt_temp.lamda,
                                                                      item_category, device, Ks)

        # 打印测试结果
        cprint(log_path_txt, f'Current Result Epoch {epoch}:')
        print_result(log_path_txt, 'Recall', recall, Ks)
        print_result(log_path_txt, 'MRR', mrr, Ks)
        print_result(log_path_txt, 'NDCG', ndcg, Ks)
        print_auc_result(log_path_txt, 'AUC', auc, Ks_auc)

    # load_optimal_model()
