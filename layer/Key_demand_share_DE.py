# -*- coding: utf-8 -*-
"""
# @File    : Key_demand_share_DE.py
# Desc:
Key 与 demand share mlp 层
"""

import math
import torch as t
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as fn

print("-----------import Key_demand_share_DE.py--------------")


class DemandExtraction(Module):
    def __init__(self, n_demand, n_categories, embedding_dim_c, hidden_size, score_act_fun=None, demand_extract='mlp',
                 demand_mask=False, demand_agg="exp"):
        """
        初始化函数
        Args:
            n_demand: int, 需求数量
            n_categories: int, 类别节点的个数
            embedding_dim_c: int, 类别嵌入到维度
            hidden_size: int, 中间隐层（中间结果层的变量维度）
            score_act_fun: str, None, relu, sigmoid
            demand_extract: str, mlp, dot
            demand_mask: bool, default=False, whether mask demand extraction part
            score_act_fun: default=None, relu, the activate function for demand score and demand score candidate
            demand_agg：str, default="exp"， "mean"
        """
        super(DemandExtraction, self).__init__()
        self.n_categories = n_categories  # include 补[0]，
        self.n_demand = n_demand
        self.hidden_size = hidden_size

        self.score_act_fun = score_act_fun
        self.demand_extract = demand_extract
        self.demand_mask = demand_mask
        self.demand_agg = demand_agg

        self.embedding_c = nn.Embedding(n_categories, embedding_dim_c, padding_idx=0)
        self.demand_linear = nn.Linear(embedding_dim_c, n_demand * hidden_size,
                                       bias=False)
        self.key_linear = nn.Linear(embedding_dim_c, hidden_size) 

        assert self.demand_extract == 'mlp' or self.demand_extract == 'dot'
        if self.demand_extract == 'mlp':
            self.w_score = nn.Parameter(torch.Tensor(self.hidden_size))
            self.nn_linear_score = nn.Sequential(
                nn.Linear(2 * self.hidden_size, self.hidden_size),
                nn.ReLU()
            )
            self.weight_init()  # for self.w_score initialization

    def weight_init(self):
        """
        初始化该layer自己定义而非子模块的参数
        """
        std = 1.0 / math.sqrt(self.hidden_size)
        self.w_score.data.uniform_(std)
        return None

    def compute_demand_score_dot(self, hidden_demand_agg, hidden_key, hidden_key_candidate):
        """
        demand score 的计算方式，和ppt中的一样
        Args:
            hidden_demand_agg: torch.Tensor, batch_size * n_demand * hidden_size
            hidden_key: torch.Tensor, batch_size * max_session_len * n_demand * hidden_size
            hidden_key_candidate: torch.Tensor, 1 * n_items * n_demand * hidden_size

        Returns:
            demand_score: torch,Tensor, batch_size  * n_demand * max_session_len
            demand_score_candidate: torch,Tensor,  batch_size * n_demand * n_items
        """
        hidden_demand_agg = hidden_demand_agg.unsqueeze(-1)  # batch_size * n_demand * hidden_size * 1
        hidden_key = hidden_key.transpose(2, 1)  # batch_size * n_demand * max_session_len * hidden_size
        demand_score = t.matmul(hidden_key, hidden_demand_agg) / (
                    self.hidden_size ** (1 / 2))  # batch_size * n_demand * max_session_len * 1
        demand_score = demand_score.squeeze(-1)  # batch_size * n_demand * max_session_len

        hidden_key_candidate = hidden_key_candidate.transpose(2, 1)  # 1*  n_demand * n_items * hidden_size
        demand_score_candidate = t.matmul(hidden_key_candidate, hidden_demand_agg) / (
                self.hidden_size ** (1 / 2))  # batch_size * n_demand * n_items * 1
        demand_score_candidate = demand_score_candidate.squeeze(-1)  # batch_size * n_demand * n_items

        return demand_score, demand_score_candidate

    def compute_demand_score_mlp(self, hidden_demand_agg, hidden_key,
                                 hidden_key_candidate):  
        """
         demand score 的计算方式，使用 mlp
         Args:
             hidden_demand_agg: torch.Tensor, batch_size * n_demand * hidden_size
             hidden_key: torch.Tensor, batch_size * max_session_len  * hidden_size
             hidden_key_candidate: torch.Tensor, n_items * hidden_size

         Returns:
             demand_score: torch,Tensor, batch_size * n_demand * max_session_len
             demand_score_candidate: torch,Tensor,  batch_size * n_demand * n_items
         """
        batch_size, max_session_len, n_items = hidden_key.shape[0], hidden_key.shape[1], hidden_key_candidate.shape[0]

        hidden_key = hidden_key.view(batch_size, max_session_len, 1, self.hidden_size).repeat(1, 1, self.n_demand, 1)
        hidden_demand_agg_1 = hidden_demand_agg.view(batch_size, 1, self.n_demand, self.hidden_size).repeat(1,
                                                                                                            max_session_len,
                                                                                                            1, 1)
        demand_score = self.nn_linear_score(torch.cat((hidden_demand_agg_1, hidden_key),
                                                      dim=-1))  # batch_size * max_session_len * n_demand * hidden_size
        demand_score = torch.matmul(demand_score, self.w_score.view(1, 1, self.hidden_size, 1)).view(batch_size,
                                                                                                     self.n_demand,
                                                                                                     max_session_len)

        hidden_key_candidate = hidden_key_candidate.view(1, 1, -1, self.hidden_size).repeat(batch_size, self.n_demand,
                                                                                            1,
                                                                                            1)  
        hidden_demand_agg = hidden_demand_agg.view(batch_size, self.n_demand, 1, self.hidden_size).repeat(1, 1, n_items,
                                                                                                          1)
        demand_score_candidate = self.nn_linear_score(torch.cat((hidden_demand_agg, hidden_key_candidate),
                                                                dim=-1))  # batch_size * n_demand * n_items * hidden_size
        demand_score_candidate = torch.matmul(demand_score_candidate,
                                              self.w_score.view(1, 1, self.hidden_size, 1)).squeeze(
            -1)  # batch_size * n_demand * n_items

        return demand_score, demand_score_candidate

    def forward(self, input, candidate_pool_category):
        """
        模型返回对应的demand score
        Args:
            input:  sess_categories_batch，torch.Tensor， dtype = torch.int64， batch_size * max_session_len
            candidate_pool_category： torch.Tensor, n_items * 1 , each row such as [category_id], the order is one-to-one mapping candidate_pool_item
        Returns:
            demand_score：torch.Tensor, dtype=torch.float32  batch_size * n_demand * max_session_len
            demand_score_candidate: torch.Tensor, dtype=torch.float32  batch_size * n_demand * n_items
            embedding: catgy embedding, torch.Tensor, dtype=torch.float32, batch_size * max_session_len * embedding_dim_c
            embedding_candidiate:torch.Tensor, dtype=torch.float32, n_items * embedding_dim_c
            demand_sim_loss: demand similarity loss torch.float64
        """
        n_items = len(candidate_pool_category)

        batch_size, max_session_len = input.shape
        embedding = self.embedding_c(input)  # batch_size * max_session_len * embedding_dim_c
        embedding_candidate = self.embedding_c(candidate_pool_category)  # n_items * embedding_dim_c
        if self.demand_mask:
            demand_score = torch.ones((batch_size, self.n_demand, max_session_len)).to(device=input.device).div(
                max_session_len)
            demand_score_candidate = torch.ones((batch_size, self.n_demand, n_items)).to(device=input.device).div(
                self.n_demand)
            # demand_score: batch_size * n_demand * max_session_len
            # demand_score_candidate: batch_size * n_demand * n_items
            return demand_score, demand_score_candidate, embedding, embedding_candidate

        hidden_key = self.demand_linear(embedding).view(batch_size, max_session_len, self.n_demand,
                                                        self.hidden_size)
        hidden_demand = self.demand_linear(embedding).view(batch_size, max_session_len, self.n_demand,
                                                           self.hidden_size)  # batch_size * max_session_len * n_demand * hidden_size 
        if self.demand_agg == "exp":
            hidden_demand_agg = hidden_demand.exp().sum(1).log()  # batch_size * n_demand * hidden_size
        elif self.demand_agg == "mean":
            hidden_demand_agg = hidden_demand.sum(1)  # 均值聚合session的需求表达
        else:
            print("opt.demand_agg value is wrong !!!!")
        demand_sim_loss = self.demand_similarity_loss(hidden_demand_agg)
        hidden_key_candidate = self.demand_linear(embedding_candidate).view(1, -1, self.n_demand,
                                                                            self.hidden_size)  # 1 * n_items * n_demand * hidden_size

        if self.demand_extract == 'dot':
            demand_score, demand_score_candidate = self.compute_demand_score_dot(hidden_demand_agg, hidden_key,
                                                                                 hidden_key_candidate)
        elif self.demand_extract == 'mlp':
            demand_score, demand_score_candidate = self.compute_demand_score_mlp(hidden_demand_agg, hidden_key,
                                                                                 hidden_key_candidate)
            # demand_score: batch_size * n_demand * max_session_len
            # demand_score_candidate: batch_size * n_demand * n_items

        if self.score_act_fun == 'relu':
            demand_score = t.relu(demand_score)  # 为了互信息的BCEWITHLOGITLOSS 的 计算，可以考虑加激活函数demand_score \in [0,1]
            demand_score_candidate = t.relu(demand_score_candidate)
        elif self.score_act_fun == 'sigmoid':
            demand_score = t.sigmoid(demand_score)
            demand_score_candidate = t.sigmoid(demand_score_candidate)
        return demand_score, demand_score_candidate, embedding, embedding_candidate, demand_sim_loss

    def demand_similarity_loss(self, demand):
        '''
        Compute demand similarity loss
        Args:
            demand: batch_size * n_demand * hidden_size

        Returns:
            Demand loss: loss
        '''
        # sim_matrix = torch.bmm(demand, demand.transpose(1, 2)) / (
        #             demand.shape[-1] ** 2)  # batch_size * n_demand * n_demand
        sim = []
        for i in range(demand.shape[1]):
            for j in range(demand.shape[1]):
                if i == j:
                    continue
                else:
                    s = torch.mean(fn.cosine_similarity(demand[:, i, :], demand[:, j, :]))
                    sim.append(s)
        loss = torch.mean(torch.stack(sim))
        return loss
