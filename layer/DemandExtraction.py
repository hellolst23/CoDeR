# -*- coding: utf-8 -*-

"""
    dec: demand extraction 部分
"""
import math
import torch as t
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as fn

class DemandExtraction(Module):
    def __init__(self, n_demand, n_categories, embedding_dim_c, hidden_size, drop=0.25, predict_catgy_fun = 'mlp',score_act_fun=None, demand_extract='mlp',
                 demand_mask=False, demand_agg="exp"):
        """
        初始化函数
        Args:
            n_demand: int, 需求数量
            n_categories: int, 类别节点的个数
            embedding_dim_c: int, 类别嵌入到维度
            hidden_size: int, 中间隐层（中间结果层的变量维度）
            drop: float, default=0.5
            predict_catgy_fun:str, prediction function: mlp
            score_act_fun: str, None, relu, sigmoid
            demand_extract: str, mlp, dot
            demand_mask: bool, default=False, whether mask demand extraction part
            score_act_fun: default=None, relu, the activate function for demand score and demand score candidate
            demand_agg：str, default="exp"， "mean"
            # visual_catgy_deamnd: default=false, case study of category demand
        """
        super(DemandExtraction, self).__init__()
        # if visual_catgy_deamnd:
        #     self.visual_catgy_demand = visual_catgy_deamnd

        self.n_categories = n_categories  # include 补[0]，
        self.n_demand = n_demand
        self.hidden_size = hidden_size

        self.score_act_fun = score_act_fun
        self.demand_extract = demand_extract
        self.demand_mask = demand_mask
        self.demand_agg = demand_agg
        self.predict_catgy_fun = predict_catgy_fun

        self.embedding_c = nn.Embedding(n_categories, embedding_dim_c, padding_idx=0)
        self.dropout = nn.Dropout(drop) # NOTE: add drop out
        self.demand_linear = nn.Linear(embedding_dim_c, n_demand * hidden_size,
                                       bias=False)
        self.key_linear = nn.Linear(embedding_dim_c, hidden_size)  # TODO: 是否要map到M个空间？

        assert self.demand_extract == 'mlp' or self.demand_extract == 'dot'
        if self.demand_extract == 'mlp':
            self.w_score = nn.Parameter(torch.Tensor(self.hidden_size))
            self.nn_linear_score = nn.Sequential(
                nn.Linear(2 * self.hidden_size, self.hidden_size),
                nn.ReLU()
            )

        if self.demand_agg == 'attention':
                self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
                self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
                self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        # self.weight_init()  # for self.w_score initialization



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
            hidden_key: torch.Tensor, batch_size * max_session_len * hidden_size
            hidden_key_candidate: torch.Tensor, n_items * hidden_size

        Returns:
            demand_score: torch,Tensor, batch_size  * n_demand * max_session_len
            demand_score_candidate: torch,Tensor,  batch_size * n_demand * n_items
        """
        batch_size, max_session_len = hidden_key.shape[0], hidden_key.shape[1]

        hidden_key = hidden_key.view(batch_size, max_session_len, 1, self.hidden_size, 1)
        hidden_demand_agg = hidden_demand_agg.view(batch_size, 1, self.n_demand, 1, self.hidden_size)
        demand_score = t.matmul(hidden_demand_agg, hidden_key) / (
                self.hidden_size ** (1 / 2))  # batch_size * max_session_len * n_demand * 1 * 1
        demand_score = demand_score.view(batch_size, self.n_demand,
                                         max_session_len)  # batch_size  * n_demand * max_session_len

        hidden_key_candidate = hidden_key_candidate.view(1, 1, -1, self.hidden_size)  # 1*  1 * n_items * hidden_size
        hidden_demand_agg = hidden_demand_agg.transpose(-1, -2).squeeze(1)  # batch_size * n_demand * hidden_size * 1
        demand_score_candidate = t.matmul(hidden_key_candidate, hidden_demand_agg) / (
                self.hidden_size ** (1 / 2))  # batch_size * n_demand * n_items * 1
        demand_score_candidate = demand_score_candidate.squeeze(-1)  # batch_size * n_demand * n_items

        return demand_score, demand_score_candidate

    def compute_demand_score_mlp(self, hidden_demand_agg, hidden_key, hidden_key_candidate):
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

    def forward(self, input, candidate_pool_category, session_last_catgy_index,mask_catgy):
        """
        模型返回对应的demand score
        Args:
            i
            demand_score：torch.Tensor, dtype=torch.float32  batch_size nput:  sess_categories_batch，torch.Tensor， dtype = torch.int64， batch_size * max_session_len
            candidate_pool_category： torch.Tensor, n_items * 1 , each row such as [category_id], the order is one-to-one mapping candidate_pool_item
        Returns:* n_demand * max_session_len
            demand_score_candidate: torch.Tensor, dtype=torch.float32  batch_size * n_demand * n_items
            embedding: catgy embedding, torch.Tensor, dtype=torch.float32, batch_size * max_session_len * embedding_dim_c
            embedding_candidiate:torch.Tensor, dtype=torch.float32, n_items * embedding_dim_c
            demand_sim_loss: demand similarity loss torch.float64
            # batch_size * max_session_len * n_demand * hidden_size
        """
        n_items = len(candidate_pool_category)

        batch_size, max_session_len = input.shape
        embedding = self.embedding_c(input)  # batch_size * max_session_len * embedding_dim_c
        embedding = self.dropout(embedding)

        embedding_candidate = self.embedding_c(candidate_pool_category)  # n_items * embedding_dim_c

        if self.demand_mask:
            demand_score = torch.ones((batch_size, self.n_demand, max_session_len)).to(device=input.device).div(max_session_len)
            demand_score_candidate = torch.ones((batch_size, self.n_demand, n_items)).to(device=input.device).div(self.n_demand)
            catgy_score = torch.ones((batch_size, self.n_demand, self.n_categories)).to(device=input.device).div(self.n_demand)
            # demand_score = torch.ones((batch_size, self.n_demand, max_session_len)).to(device=input.device)
            # demand_score_candidate = torch.ones((batch_size, self.n_demand, n_items)).to(device=input.device) # NOTE: 设置为1

            # demand_score: batch_size * n_demand * max_session_len
            # demand_score_candidate: batch_size * n_demand * n_items
            # catgy_score: batch_size * n_demand * n_categoryies
            return catgy_score, demand_score, demand_score_candidate, embedding, embedding_candidate, torch.tensor([0]).float().to(embedding_candidate.device)

        hidden_key = self.key_linear(embedding)  # batch_size * max_session_len * embedding_dim_c
        # hidden_key = torch.selu(hidden_key)
        hidden_demand = self.demand_linear(embedding).view(batch_size, max_session_len, self.n_demand,
                                                           self.hidden_size)  # batch_size * max_session_len * n_demand * hidden_size # TODO: 是否可以加上relu等非线性激活函数？
        # hidden_demand = torch.selu(hidden_demand)
        if self.demand_agg == "exp":
            hidden_demand_agg = hidden_demand.exp().sum(1).log()  # batch_size * n_demand * hidden_size
        elif self.demand_agg == "mean":
            hidden_demand_agg = hidden_demand.sum(1)  # 均值聚合session的需求表达
        elif self.demand_agg == "attention": # attention 聚合
            hidden_demand_agg = self.att_demand(hidden_demand, session_last_catgy_index,mask_catgy)
        else:
            print("opt.demand_agg value is wrong !!!!")
        demand_sim_loss = self.demand_similarity_loss(hidden_demand_agg)
        hidden_key_candidate = self.key_linear(embedding_candidate)  # n_items * hidden_size
        # hidden_key_candidate = torch.selu(hidden_key_candidate)

        if self.demand_extract == 'dot':
            demand_score, demand_score_candidate = self.compute_demand_score_dot(hidden_demand_agg, hidden_key,
                                                                                 hidden_key_candidate)
        elif self.demand_extract == 'mlp':
            demand_score, demand_score_candidate = self.compute_demand_score_mlp(hidden_demand_agg, hidden_key,
                                                                                 hidden_key_candidate)
        # demand_score: batch_size * n_demand * max_session_len
        # demand_score_candidate: batch_size * n_demand * n_items

        if self.score_act_fun == 'relu':
            demand_score = t.relu(demand_score)
            demand_score_candidate = t.relu(demand_score_candidate)
        elif self.score_act_fun == 'sigmoid':
            demand_score = t.sigmoid(demand_score)
            demand_score_candidate = t.sigmoid(demand_score_candidate)
        catgy_score = self.predict_catgy(hidden_demand_agg)
        return catgy_score, demand_score, demand_score_candidate, embedding, embedding_candidate, demand_sim_loss

    def att_demand(self,hidden_demand, session_last_catgy_index, mask_catgy):
        """
        attention aggregation for session demand
        Args:
            hidden_demand: batch_size * max_session_len * n_demand * hidden_size
            mask_nodes: torch.Tensor, dtype=torch.int64, batch_size * max_nodes_len, record the clicked nodes in a session

        Returns:

        """
        batch_size, max_nodes_len, n_demand, hidden_size = hidden_demand.shape
        last_catgy_info = hidden_demand[torch.arange(
            len(session_last_catgy_index)), session_last_catgy_index]  # batch_size * n_demand * embedding_dim_node
        q1 = self.linear_one(last_catgy_info).unsqueeze(2)  # batch_size * n_demand * 1 * embedding_dim_node
        q2 = self.linear_two(hidden_demand.transpose(2,1))  # batch_size   * n_demand  * max_nodes_len * embedding_dim_node
        alpha = self.linear_three(torch.sigmoid(q1 + q2))  # batch_size * n_demand * max_nodes_len * 1
        hidden_demand = alpha * hidden_demand.transpose(2,1)  # batch_size * n_demand * max_nodes_len * embedding_dim_node
        hidden_demand_agg = torch.matmul(mask_catgy.view(batch_size, 1, 1, max_nodes_len).float(), hidden_demand).squeeze(
            -2)  # batch_size * n_demand * embedding_dim_node
        return hidden_demand_agg
        """
        # hidden: batch_size * n_demand * max_nodes_len * embedding_dim_node,
        last_catgy_info = hidden.transpose(2, 1)[torch.arange(
            len(session_last_item_index)), session_last_item_index]  # batch_size * n_demand * embedding_dim_node
        q1 = self.linear_one(last_item_info).unsqueeze(2)  # batch_size * n_demand * 1 * embedding_dim_node
        q2 = self.linear_two(hidden)  # batch_size * n_demand * max_nodes_len * embedding_dim_node
        alpha = self.linear_three(torch.sigmoid(q1 + q2))  # batch_size * n_demand * max_nodes_len * 1

        hidden = alpha * hidden  # batch_size * n_demand * max_nodes_len * embedding_dim_node
        a = torch.matmul(mask_node.view(batch_size, 1, 1, max_nodes_len).float(), hidden).squeeze(
            -2)  # batch_size * n_demand * embedding_dim_node
        graph_representation = a
        """
        """
        graph_representation = torch.matmul(mask_nodes.view(batch_size, 1, 1, max_nodes_len), gnn_result).squeeze(
            -2)  # batch_size * n_demand * embedding_dim_node
        graph_representation = graph_representation.div(mask_nodes.sum(-1).view(batch_size, 1, 1))
        """

    def predict_catgy(self, hidden_demand_agg):
        """

        :param hidden_demand_agg: torch.tensor, batch_size * n_demand * hidden_size
        :return: score_c, torch.tensor, batch_size * n_demand * n_categoryies
        """
        batch_size, n_demand, _ = hidden_demand_agg.shape
        catgy_candidate = self.embedding_c.weight[0:]  # n_categories * hidden_size
        # catgy_candidate = F.normalize(catgy_candidate, p=2, dim=-1)
        # catgy_candidate = max_norm_fun(catgy_candidate, max_norm=1, mode='l2')
        if self.predict_catgy_fun == 'mlp':
            temp = torch.cat([hidden_demand_agg.unsqueeze(-2).repeat(1, 1, self.n_categories, 1),
                              catgy_candidate.unsqueeze(0).unsqueeze(0).repeat(batch_size, n_demand, 1, 1)], dim=-1)
            score_c = self.catgy_linear(temp)
            score_c = torch.sigmoid(score_c.squeeze(-1))
        elif self.predict_catgy_fun == 'dot':
            score_c = t.matmul(hidden_demand_agg,
                               catgy_candidate.transpose(1, 0).unsqueeze(0))  # batch_size * n_demand * n_categories
        # score_c = score_c.max(1).values
        return score_c

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
        device = demand.device
        if (demand.shape[1] == 1):
            return torch.zeros(1).to(device)
        for i in range(demand.shape[1]):
            for j in range(demand.shape[1]):
                if i == j:
                    continue
                else:
                    s = torch.relu(torch.mean(fn.cosine_similarity(demand[:, i, :], demand[:, j, :])))
                    sim.append(s)
        loss = torch.mean(torch.stack(sim))
        return loss

# 按照transformer的实现重新更改的demand extraction模块

# class DemandExtraction(Module):

#     def __init__(self, n_demand, n_categories, embedding_dim_c, hidden_size, score_act_fun=None, demand_extract='mlp',
#                  demand_mask=False, demand_agg="exp"):
#         """
#         按照transformer的实现重新更改的demand extraction模块
#         初始化函数
#         Args:
#             n_demand: int, 需求数量
#             n_categories: int, 类别节点的个数
#             embedding_dim_c: int, 类别嵌入到维度
#             hidden_size: int, 中间隐层（中间结果层的变量维度）
#             score_act_fun: str, None, relu, sigmoid
#             demand_extract: str, mlp, dot
#             demand_mask: bool, default=False, whether mask demand extraction part
#             score_act_fun: default=None, relu, the activate function for demand score and demand score candidate
#             demand_agg：str, default="exp"， "mean"
#         """
#         super(DemandExtraction, self).__init__()
#         self.n_categories = n_categories  # include 补[0]，
#         self.n_demand = n_demand
#         self.hidden_size = hidden_size

#         self.score_act_fun = score_act_fun
#         self.demand_extract = demand_extract
#         self.demand_mask = demand_mask
#         self.demand_agg = demand_agg

#         self.embedding_c = nn.Embedding(n_categories, embedding_dim_c, padding_idx=0)
#         assert hidden_size % n_demand == 0, "hidden_size  % n_demand != 0"
#         self.hidden_demand_size = int(self.hidden_size / self.n_demand)
#         self.demand_linear = nn.Linear(embedding_dim_c, hidden_size,
#                                        bias=False)
#         self.key_linear = nn.Linear(embedding_dim_c, hidden_size)

#         assert self.demand_extract == 'mlp' or self.demand_extract == 'dot'
#         if self.demand_extract == 'mlp':
#             self.w_score = nn.Parameter(torch.Tensor(self.hidden_size))
#             self.nn_linear_score = nn.Sequential(
#                 nn.Linear(self.hidden_size, self.hidden_size),
#                 nn.ReLU()
#             )
#             self.weight_init()  # for self.w_score initialization

#     def weight_init(self):
#         """
#         初始化该layer自己定义而非子模块的参数
#         """
#         std = 1.0 / math.sqrt(self.hidden_size)
#         self.w_score.data.uniform_(std)
#         return None

#     def compute_demand_score_dot(self, hidden_demand_agg, hidden_key, hidden_key_candidate): # todo: 明天写计算demand_score 的方式
#         """
#         demand score 的计算方式，和ppt中的一样
#         Args:
#             hidden_demand_agg: torch.Tensor, batch_size * n_demand * hidden_demand_size
#             hidden_key: torch.Tensor, batch_size * max_session_len * n_demand * hidden_demand_size
#             hidden_key_candidate: torch.Tensor, n_items * n_demand * hidden_demand_size

#         Returns:
#             demand_score: torch,Tensor, batch_size  * n_demand * max_session_len
#             demand_score_candidate: torch,Tensor,  batch_size * n_demand * n_items
#         """
#         batch_size, max_session_len = hidden_key.shape[0], hidden_key.shape[1]

#         hidden_key = hidden_key.view(batch_size, max_session_len, self.n_demand, self.hidden_demand_size, 1)
#         hidden_demand_agg = hidden_demand_agg.view(batch_size, 1, self.n_demand, 1, self.hidden_demand_size)
#         demand_score = t.matmul(hidden_demand_agg, hidden_key) / (
#                 self.hidden_demand_size ** (1 / 2))  # batch_size * max_session_len * n_demand * 1 * 1
#         demand_score = demand_score.view(batch_size, self.n_demand,
#                                          max_session_len)  # batch_size  * n_demand * max_session_len

#         hidden_key_candidate = hidden_key_candidate.view(1, self.n_demand, -1, self.hidden_demand_size)  # 1*  n_demand * n_items * hidden_demand_size
#         hidden_demand_agg = hidden_demand_agg.transpose(-1, -2).squeeze(1)  # batch_size * n_demand * hidden_demand_size * 1
#         demand_score_candidate = t.matmul(hidden_key_candidate, hidden_demand_agg) / (
#                 self.hidden_demand_size ** (1 / 2))  # batch_size * n_demand * n_items * 1
#         demand_score_candidate = demand_score_candidate.squeeze(-1)  # batch_size * n_demand * n_items

#         return demand_score, demand_score_candidate

#     def compute_demand_score_mlp(self, hidden_demand_agg, hidden_key, hidden_key_candidate):
#         """
#          demand score 的计算方式，使用 mlp
#          Args:
#              hidden_demand_agg: torch.Tensor, batch_size * n_demand * hidden_size
#              hidden_key: torch.Tensor, batch_size * max_session_len  * hidden_size
#              hidden_key_candidate: torch.Tensor, n_items * hidden_size

#          Returns:
#              demand_score: torch,Tensor, batch_size * n_demand * max_session_len
#              demand_score_candidate: torch,Tensor,  batch_size * n_demand * n_items
#          """
#         batch_size, max_session_len, n_items = hidden_key.shape[0], hidden_key.shape[1], hidden_key_candidate.shape[0]

#         hidden_key = hidden_key.view(batch_size, max_session_len, 1, self.hidden_size).repeat(1, 1, self.n_demand, 1)
#         hidden_demand_agg_1 = hidden_demand_agg.view(batch_size, 1, self.n_demand, self.hidden_size).repeat(1,
#                                                                                                             max_session_len,
#                                                                                                             1, 1)
#         demand_score = self.nn_linear_score(torch.cat((hidden_demand_agg_1, hidden_key),
#                                                       dim=-1))  # batch_size * max_session_len * n_demand * hidden_size
#         demand_score = torch.matmul(demand_score, self.w_score.view(1, 1, self.hidden_size, 1)).view(batch_size,
#                                                                                                      self.n_demand,
#                                                                                                      max_session_len)

#         hidden_key_candidate = hidden_key_candidate.view(1, 1, -1, self.hidden_size).repeat(batch_size, self.n_demand,
#                                                                                             1,
#                                                                                             1)  # todo 醒来检查这里demand——score 的计算方式
#         hidden_demand_agg = hidden_demand_agg.view(batch_size, self.n_demand, 1, self.hidden_size).repeat(1, 1, n_items,
#                                                                                                           1)
#         demand_score_candidate = self.nn_linear_score(torch.cat((hidden_demand_agg, hidden_key_candidate),
#                                                                 dim=-1))  # batch_size * n_demand * n_items * hidden_size
#         demand_score_candidate = torch.matmul(demand_score_candidate,
#                                               self.w_score.view(1, 1, self.hidden_size, 1)).squeeze(
#             -1)  # batch_size * n_demand * n_items

#         return demand_score, demand_score_candidate

#     def forward(self, input, candidate_pool_category):
#         """
#         模型返回对应的demand score
#         Args:
#             input:  sess_categories_batch，torch.Tensor， dtype = torch.int64， batch_size * max_session_len
#             candidate_pool_category： torch.Tensor, n_items * 1 , each row such as [category_id], the order is one-to-one mapping candidate_pool_item
#         Returns:
#             demand_score：torch.Tensor, dtype=torch.float32  batch_size * n_demand * max_session_len
#             demand_score_candidate: torch.Tensor, dtype=torch.float32  batch_size * n_demand * n_items
#             embedding: catgy embedding, torch.Tensor, dtype=torch.float32, batch_size * max_session_len * embedding_dim_c
#             embedding_candidiate:torch.Tensor, dtype=torch.float32, n_items * embedding_dim_c
#             demand_sim_loss: demand similarity loss torch.float64
#         """
#         n_items = len(candidate_pool_category)


#         batch_size, max_session_len = input.shape
#         embedding = self.embedding_c(input)  # batch_size * max_session_len * embedding_dim_c
#         embedding_candidate = self.embedding_c(candidate_pool_category)  # n_items * embedding_dim_c

#         if self.demand_mask:
#             demand_score = torch.ones((batch_size, self.n_demand, max_session_len)).to(device=input.device).div(
#                 max_session_len)
#             demand_score_candidate = torch.ones((batch_size, self.n_demand, n_items)).to(device=input.device).div(
#                 self.n_demand)
#             # demand_score: batch_size * n_demand * max_session_len
#             # demand_score_candidate: batch_size * n_demand * n_items
#             return demand_score, demand_score_candidate, embedding, embedding_candidate, torch.tensor([0]).float().to(embedding_candidate.device)

#         hidden_key = self.key_linear(embedding).view(batch_size, max_session_len, self.n_demand,
#                                                            self.hidden_demand_size)  # batch_size * max_session_len *n_demand * hidden_demand_size
#         hidden_demand = self.demand_linear(embedding).view(batch_size, max_session_len, self.n_demand,
#                                                            self.hidden_demand_size)  # batch_size * max_session_len * n_demand * hidden_demand_size #
#         if self.demand_agg == "exp":
#             hidden_demand_agg = hidden_demand.exp().sum(1).log()  # batch_size * n_demand * hidden_demand_size
#         elif self.demand_agg == "mean":
#             hidden_demand_agg = hidden_demand.sum(1)  # 均值聚合session的需求表达
#         else:
#             print("opt.demand_agg value is wrong !!!!")
#         demand_sim_loss = self.demand_similarity_loss(hidden_demand_agg)
#         hidden_key_candidate = self.key_linear(embedding_candidate).view(n_items, self.n_demand, self.hidden_demand_size)  # n_items * n_demand * hidden_demand_size

#         if self.demand_extract == 'dot':
#             demand_score, demand_score_candidate = self.compute_demand_score_dot(hidden_demand_agg, hidden_key,
#                                                                                  hidden_key_candidate)
#         # demand_score: batch_size * n_demand * max_session_len
#         # demand_score_candidate: batch_size * n_demand * n_items

#         if self.score_act_fun == 'relu':
#             demand_score = t.relu(demand_score)  # 为了互信息的BCEWITHLOGITLOSS 的 计算，可以考虑加激活函数demand_score \in [0,1]
#             demand_score_candidate = t.relu(demand_score_candidate)
#         elif self.score_act_fun == 'sigmoid':
#             demand_score = t.sigmoid(demand_score)
#             demand_score_candidate = t.sigmoid(demand_score_candidate)
#         return demand_score, demand_score_candidate, embedding, embedding_candidate, demand_sim_loss

#     def demand_similarity_loss(self, demand):
#         '''
#         Compute demand similarity loss
#         Args:
#             demand: batch_size * n_demand * hidden_size

#         Returns:
#             Demand loss: loss
#         '''
#         # sim_matrix = torch.bmm(demand, demand.transpose(1, 2)) / (
#         #             demand.shape[-1] ** 2)  # batch_size * n_demand * n_demand
#         sim = []
#         device = demand.device
#         if (demand.shape[1] == 1):
#             return torch.zeros(1).to(device)
#         for i in range(demand.shape[1]):
#