#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : PVSD.py
# Desc:
calculate P(v: s, d) in RS part
"""

from layer import *
from util.utils import timefn

# BatchNormal 批量正则化 映射到均值为 0 ，方差为 1 的正态分布
class PVSD(nn.Module): 
    def __init__(self, hidden_size, batchNorm = 'feature', n_demand=2, rs='dot'):
        """
        Creat P_v_s_d class to add batchNormal layer conveniently.
        Args:
            hidden_size: int,  nn.Linear weight shape
            batchNorm' opt.batchNorm, str, demand, feature, None
            n_demand: opt.n_demand, int, such as 2, 3, 4, ...
            rs: str, 'dot' or 'mlp' in recommendation part, 'dot': 点积
        """
        super(PVSD, self).__init__()
        self.batchNorm = batchNorm
        self.rs = rs
        self.hidden_size = hidden_size

        if self.batchNorm == 'demand':
            self.bn = nn.BatchNorm2d(n_demand)
        elif self.batchNorm == 'feature':
            self.bn = nn.BatchNorm1d(hidden_size)
        else:
            self.bn = None

        self.act_func = nn.ReLU()

        assert self.rs == 'mlp' or self.rs == 'dot'
        if self.rs == 'mlp':
            self.nn_linear1 = nn.Linear(3*hidden_size, hidden_size)
            #self.nn_linear2 = Linear2D(hidden_size, 1)
            self.nn_linear2 = nn.Linear(hidden_size,1)

        elif self.rs == 'dot':
            self.nn_linear1 = nn.Linear(2*hidden_size, hidden_size)

    def batch_norm_custom(self, hidden):
        """
       batchNorm on hidden, do not change hidden shape
        Args:
            hidden: torch.Tensor, batch_size * n_demand * (*) *  hidden_size , (*) represent any size
        Returns:
            hidden: torch.Tensor, batch_size * n_demand * (*) *  hidden_size , (*) represent any size
        """
        if self.batchNorm == 'demand':
            hidden = self.bn(hidden)
        elif self.batchNorm == 'feature':
            assert len(hidden.shape) == 4
            batch_size, n_demand, _, hidden_size1 = hidden.shape
            hidden = hidden.view(-1, hidden_size1)
            hidden = self.bn(hidden)
            hidden = hidden.view(batch_size, n_demand, _, hidden_size1)
        return hidden

    def forward(self, sess_represenation, candidate_items_embedding):
        """

        Args:
            sess_represenation: torch.Tensor, # batch_size * n_demand * (embedding_dim_node * 2)
            candidate_items_embedding: torch.Tensor, batch_size * n_demand * (n_item+1) *  hidden_size
        Returns:
            p_v_s_d: torch.Tensor, batch_size * n_demand * (n_item+1)
        """
        # Firstly, batchNorm, then action function
        if self.rs == 'mlp':
            sess_represenation = sess_represenation.unsqueeze(2).repeat(1, 1, candidate_items_embedding.shape[2], 1)
            # Batch_size * n_demand * (n_item+1) * (embedding_dim_node * 2)
            hidden = torch.cat((sess_represenation, candidate_items_embedding), dim=-1)
            hidden = self.nn_linear1(hidden)  # batch_size * n_demand * (n_item+1) *  hidden_size
            hidden = self.batch_norm_custom(hidden)  # batch_size * n_demand * (n_item+1) *  hidden_size
            p_v_s_d = self.nn_linear2(self.act_func(hidden)).squeeze(-1)  # batch_size * n_demand * (n_item+1)
            # todo 这里考虑是否要加一个激活函数 Relu，然后再到loss中经过softmax

        elif self.rs == 'dot':
            """
            20210328 之前的版本， 加last item
            hidden = self.nn_linear1(sess_represenation)  # batch_size * n_demand * hidden_size
            hidden = hidden.unsqueeze(-2)  # batch_size * n_demand * 1 * hidden_size
            hidden = self.batch_norm_custom(hidden)  # batch_size * n_demand * 1 * hidden_size
            hidden = self.act_func(hidden) # batch_size * n_demand * 1 * hidden_size
            # candidate_items_embedding: batch_size * n_demand * (n_item+1) *  hidden_size
            p_v_s_d = torch.matmul(candidate_items_embedding, hidden.transpose(-1,-2)).squeeze(-1)  # batch_size * n_demand * (n_item+1)
            # todo 这里考虑是否要加一个激活函数 Relu，然后再到loss中经过softmax
            """

            hidden = sess_represenation[:,:,:self.hidden_size]  # batch_size * n_demand * hidden_size
            hidden = hidden.unsqueeze(-2)  # batch_size * n_demand * 1 * hidden_size
            # candidate_items_embedding: batch_size * n_demand * (n_item+1) *  hidden_size
            p_v_s_d = torch.matmul(candidate_items_embedding, hidden.transpose(-1, -2)).squeeze(
                -1)  # batch_size * n_demand * (n_item+1)
        return p_v_s_d