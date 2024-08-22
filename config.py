#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : config.py
# Dec: Parameters for running experiments seriously

import warnings
import torch as t
from util.utils import file_write




class DefaultConfig(object):

    beizhu ="datasets:digi, RS: SR-GNN, last item,  MeanGNN, key demand share"
    # beizhu ="datasets:tmall，key demand share，hidden_size=100，sr_gnn"
    beizhu1 = "info"

    define_dictory = "digi_demand_agg_exp"
    log_path_txt = 'output.txt'
    seed = 2024

    # the gap modularity
    alpha = 0.9

    # base
    gpu_id = 0  # default='0', help='gpu id'
    # dataset = "tafeng"
    # dataset = "tmall"
    dataset = "diginetica"
    epoch = 50 # type=int, default=100, help='the number of epochs to train for' digi epoch = 15
    batch_size = 128  # type=int, default=100, help='input batch size'#todo


    hidden_size = 100  # type=int, default=100, help='hidden state size'
    embedding_dim_c = hidden_size  # type=int, default=100, help='category hidden state size'
    embedding_dim_i = hidden_size  # type=int, default=100, help='item hidden state size'
    lamda = 0.1 # type=float, default=0.1, help='hyper-parameter, to balance the impact between info_max loss and click loss'


    demand_mask = True # bool, default= False
    gnn_mask = False  # bool, default=False
    recommend_model = "sr_gnn"  # str, 'sr_gnn', 'demandRs', default=None


    n_demand = 2  # type=int, default=2, help='the number of demands in a session'
    dashed_order = 2  # type=int, default=2, help='the size of sliding window for construction the dashed adjacent matrix '
    n_gnn_layer = 1  # type=int, default=1, help='the number of gnn layer'
    bias = True  # action='store_false', help='bias in GNN linear layer '
    non_linear = 'relu'  # type=str, default='relu', help='the activation function for GNN output layer'
    non_linear_demand_score = 'sigmoid'  # type=str, default='sigmoid', help='the activation function for demand score'
    batch_norm = 'feature'  # type=str, default='feature', help='None/demand/feature ... , which elements should be normalized'


    add_pos = False  # type=bool， default=False, help="whether pos embedding is added into item embedding

    catgy_embedding = True  # type=bool， default=False, help="whether catgy embedding is passed into GCN part
    dashed_weight = "log"  #  type=str, default='log', help='log, log_exp, no_log, the method of compute dashed edge weight'
    softmax_demand_score_candidate = False  # type=bool， default=False, help="whether demand_score_candidate is passed into softmax"


    graph_aggregation = 'mean'  #  type=str, default='mean', help='method to aggregate the graph, lstm, mean,sum...'
    rs = "dot"  # type=str, default='dot', help='dot/mlp, p_v_s_d calculation method in RS part

    demand_agg = "exp"  # str, default="exp"， "mean","attention" help="the way of demand aggregation" # todo tmall

    demand_extract = 'dot'  # type=str, default='mlp', help='dot/mlp, demand score calculate method in demand extraction part'
    demand_share_agg = True  # type=bool, default=True, help='Whether neighbor aggregation parameters in GNN are shared'
    demand_share_node = False  # type=bool, default=False, help='Whether node updating parameters in GNN are shared'

    predict_catgy_fun = 'dot' #type=str, default='mlp', help='dot/mlp, predict catgy score in demand extraction part'
    nonhybrid = False  # type=bool, default=False, help='only use the global preference to predict'

    node_out = False  # type=bool, default=False, help='aggregate neighbors'
    topk = 20  # type=int, default=20, help='recall @k, mrr@k'


    info_lamda = 1.0    # type=float, default=0.1, help='hyper-parameter, to balance the impact between info_max loss and click loss'# todo tmall 1.0
    catgy_lamda = 0.0 # type=float, default=0.5, help="hyper-parameter to balance catgy loss and rec loss")# todo
    drop_catgy = 0.5  # TMALL 0.5
    drop_item = 0.5

    loss_name = 'cross'  # type=str, default='cross',  help="the type of loss function, such as 'cross'(crossEntropy),'bpr' "
    sample_strategy = 'random'  # type=str 'random', 'category', help = "sample strategy for negative samples"
    n_negative = 3  # type=int, default=0, help='the negative samples in Loss function if n_negative=0, it represents not carrying negative strategy'
    beta = 1.0  # type=float, default=1.0, help='hyper-parameter to balance negative samples and target sample in loss funciton'
    embed_l2 = 0  # 1e-5 # type=float, default=0.0, help='l2 penalty' such as [0.001, 0.0005, 0.0001, 0.00005, 0.00001]


    lr = 0.001  # type=float, default=0.001, help='learning rate' such as [0.001, 0.0005, 0.0001] # todo tafeng 0.001
    lr_dc = 0.1  # type=float, default=0.1, help='learning rate decay rate'

    lr_dc_step = 1 # type=int, default=3, help='the number of steps after which the learning rate decay'

    l2 = 1e-5  # type=float, default=1e-5, help='l2 penalty such as [0.001, 0.0005, 0.0001, 0.00005, 0.00001], it is not used.'
    patience = 5  # type=int, default=20, help='the number of epoch to wait before early stop '
    validation = False  # action='store_true', help='validation' such as action='store_true' 默认值未false # todo
    valid_portion = 0.1 # type=float, default=0.1, help='split the portion of training set as validation set' # todo tafeng 0.5


    comment = ''  # type=str, default='', help='visualization comment in tensorboard'

    # others
    n_node = 0
    n_node_c = 0


    #max_num_nodes = 50  # type=int, default=50, help='the max number of unique nodes in a session'

    file_write(log_path_txt, '------------import config.py---------------')

    def parse(self):

        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')
        """
        file_write(self.log_path_txt, 'user config:')
        parameters= []
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_') and k not in ('n_node', 'n_node_c'):
                parameters.append((k, getattr(self, k)))
        parameters_str = ''.join(str(e) for e in parameters)
        file_write(self.log_path_txt, parameters_str)


opt1 = DefaultConfig()

