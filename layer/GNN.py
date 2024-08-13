#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch as t
from torch import nn
from torch.nn import Module
from layer.Linear3D import Linear3D


class GNN(Module):
    def __init__(self,config, n_demand, embedding_dim_node, dashed_order, bias=True, non_linear='relu', non_linear_demand_score='sigmoid', demand_share_agg=False,demand_share_node=False, node_out=True):
        super(GNN, self).__init__()
        '''
        Args:
            n_demand: number of the demand
            embedding_dim_node: embedding dimension of the node
            dashed_order: order of the dashed edges, 1 means only the solid edges are available
            demand_share_agg: bool, demand 的在gcn 邻居聚合时，是否共享参数
            demand_share_node: bool, demand的节点更新时，是否共享参数
            node_out: bool, 聚合是否考虑出度的邻居
        '''
        self.n_demand = n_demand
        self.dashed_order = dashed_order
        self.non_linear = non_linear
        self.non_linear_demand_score = non_linear_demand_score
        self.node_out=node_out
        self.dashed_weight = config.dashed_weight

        if demand_share_agg:
            # 不同的demand，共享邻居聚合层参数
            self.solid_nn_linear = nn.Sequential(
                nn.Linear((1+self.node_out)*embedding_dim_node, embedding_dim_node, bias=bias),
                nn.ReLU()
            )
            # 虚边邻居聚合，不同的order ，共享nn_linear 层
            self.dashed_nn_linear = nn.Sequential(
                nn.Linear((1+self.node_out)*embedding_dim_node, embedding_dim_node,bias=bias),
                nn.ReLU()
            )
        else:
            # 不同的demand，不共享邻居聚合层参数
            self.solid_nn_linear = nn.Sequential(
                Linear3D(self.n_demand, (1+self.node_out)*embedding_dim_node, embedding_dim_node, bias=bias),
                nn.ReLU()
            )
            # 虚边邻居聚合，相同demand，不同的order ，共享nn_linear 层
            self.dashed_nn_linear = nn.Sequential(
                Linear3D(self.n_demand, (1+self.node_out)*embedding_dim_node, embedding_dim_node, bias=bias),
                nn.ReLU()
            )

        if demand_share_node:
            self.gnn_weight = nn.Linear((dashed_order+1)*embedding_dim_node, embedding_dim_node,bias=bias)
        else:
            self.gnn_weight = Linear3D(self.n_demand, (dashed_order+1)*embedding_dim_node, embedding_dim_node, bias=bias) # demand 更新参数不共享

        # Discriminator
        self.f_k = nn.Bilinear(embedding_dim_node, embedding_dim_node, 1)
        self.loss = nn.BCEWithLogitsLoss()
        self.weight_init()

    def weight_init(self):
        '''
        初始化该layer自己定义而非子模块的参数
        '''
        # raise NotImplementedError
        return None

    def get_node_weight(self, nodes_categories_matrixes, demand_scores):
        """
        Get node emb weight from the demand scores
        Args:
            nodes_categories_matrixes: batch_size * max_nodes_len* max_session_len
            demand_scores: torch.tensor, batch_size * n_demand * max_session_len
        Returns:
            nodes_weight: torch.tensor, batch_size * n_demand * max_nodes_len * 1
        """
        # nodes_categories_matrixes 按行归一化
        nodes_categories_matrixes = self.normalize_matrix(
            nodes_categories_matrixes)  # batch_size * max_nodes_len* max_session_len
        nodes_categories_matrixes = nodes_categories_matrixes.unsqueeze(
            1)  # batch_size * 1 * max_nodes_len* max_session_len
        demand_scores = demand_scores.unsqueeze(-1)  # batch_size * n_demand * max_session_len * 1
        nodes_weight = nodes_categories_matrixes.matmul(demand_scores)  # batch_size *n_demand* max_nodes_len * 1

        return nodes_weight

    def normalize_matrix(self, matrix):
        """
        Normalize matrix by row
        Args:
            matrix: batch_size * max_nodes_len * max_session_len,   #target * source
        Returns:
        """
        rowsum = matrix.sum(2).clamp(1, matrix.shape[-1])
        matrix = matrix / rowsum.unsqueeze(-1).float()
        return matrix

    def forward(self, demand_session_node, adj_matrixes, demand_scores, nodes_categories_matrixes,
                **kwargs):
        """
        Args:
            demand_session_node: batch_size * n_demand * max_nodes_len * embedding_dim_node
            adj_matrixes: adjacent matrix, batch_size * max_nodes_len * max_nodes_len
            demand_scores: batch_size * n_demand * max_session_len
            nodes_categories_matrixes: batch_size * max_nodes_len* max_session_len
            **kwargs:
        Returns:
            output: batch_size * n_demand * max_nodes_len * embedding_dim_node
        """
        batch_size, n_demand, max_nodes_len, embedding_dim_node = demand_session_node.shape

        # Compute nodes weight
        nodes_weight = self.get_node_weight(nodes_categories_matrixes, demand_scores)  # batch_size * n_demand * max_nodes_len * 1

        weight_demand_emb = nodes_weight * demand_session_node  # Bind weight to demand node emb, batch_size * n_demand * max_nodes_len * embedding_dim_node
        order_emb_output = []  # GNN output for each order
        adj_matrixes = adj_matrixes.permute(0, 2, 1).float()  # adj: target * source， a_ij: j->i
        k_order_matrix = adj_matrixes  # 原始邻接矩阵，存存储邻接关系，batch_size * max_nodes_len * max_nodes_len, 从行看，表示入度，列看表示出度
        # obtain dashed_edges_weight and 邻居节点表示之和
        for i in range(self.dashed_order):
            # Solid edges
            if i == 0:
                normalize_k_matrix_in = self.normalize_matrix(k_order_matrix)  # 按照节点的入度正则, batch_size * max_node_len * max_node_len
                gnn_output = t.matmul(normalize_k_matrix_in.unsqueeze(1),weight_demand_emb)
                # 计算入度的节点邻居, batch_size * n_demand * max_nodes_len * embedding_dim_node
                # Note: 由于k_order_matrix已经按照入度正则，所以不需要再进行平均
                if self.node_out:
                    k_order_matrix_out = k_order_matrix.transpose(2, 1)
                    normalize_k_matrix_out = self.normalize_matrix(k_order_matrix_out) # 按照节点的出度正则, batch_size * max_node_len * max_node_len
                    gnn_output_out = t.matmul(normalize_k_matrix_out.unsqueeze(1), weight_demand_emb) # batch_size * n_demand * max_nodes_len * embedding_dim_node
                    gnn_output = torch.cat((gnn_output, gnn_output_out), dim=-1) # batch_size * n_demand * max_nodes_len * (embedding_dim_node*2)

                gnn_output = self.solid_nn_linear(gnn_output) # batch_size * n_demand * max_nodes_len * embedding_dim_node
            # Dashed edges
            else:
                k_order_matrix = (k_order_matrix @ adj_matrixes).clamp(0, 1)
                normalize_k_matrix = self.normalize_matrix(k_order_matrix)
                k_demand_matrix = []

                if self.node_out:
                    normalize_k_matrix_out = self.normalize_matrix(k_order_matrix.transpose(2,1))
                    k_demand_matrix_out = []

                for j in range(self.n_demand):
                    if self.dashed_weight == "log_exp":
                        dashed_edges_weight = t.log(t.matmul(nodes_weight[:, j, :, :],
                                                         nodes_weight[:, j, :, :].exp().permute(0, 2,
                                                                                          1)) + 1)  # Batch_size * max_nodes_len * max_nodes_len
                    elif self.dashed_weight == "log":
                        dashed_edges_weight = t.log(t.matmul(nodes_weight[:, j, :, :],
                                                             nodes_weight[:, j, :, :].permute(0, 2,
                                                                                                    1)) + 1)  # Batch_size * max_nodes_len * max_nodes_len
                    demand_matrix = dashed_edges_weight * normalize_k_matrix
                    k_demand_matrix.append(demand_matrix)
                    if self.node_out:
                        demand_matrix_out = dashed_edges_weight.transpose(2,1) * normalize_k_matrix_out
                        k_demand_matrix_out.append(demand_matrix_out)

                k_demand_matrix = t.stack(k_demand_matrix,
                                          dim=1)  # batch_size * n_demand * max_nodes_len * max_nodes_len
                gnn_output = t.matmul(k_demand_matrix,
                                      weight_demand_emb)  # batch_size * n_demand * max_nodes_len * embedding_dim_node
                if self.node_out:
                    k_demand_matrix_out = torch.stack(k_demand_matrix_out,
                                          dim=1)  # batch_size * n_demand * max_nodes_len * max_nodes_len
                    gnn_output_out = t.matmul(k_demand_matrix_out,
                                      weight_demand_emb)  # batch_size * n_demand * max_nodes_len * embedding_dim_node
                    gnn_output = torch.cat((gnn_output, gnn_output_out), dim=-1) # batch_size * n_demand * max_nodes_len * (embedding_dim_node*2)

                gnn_output = self.dashed_nn_linear(gnn_output) # batch_size * n_demand * max_nodes_len * embedding_dim_node
            order_emb_output.append(gnn_output)
        order_emb_output = torch.cat(order_emb_output, dim=-1) # batch_size * n_demand * max_nodes_len * (embedding_dim_node * order)

        # 节点更新
        output = self.gnn_weight(torch.cat((demand_session_node, order_emb_output), dim=-1)) # batch_size * n_demand * max_nodes_len * embedding_dim_node

        if self.non_linear == 'relu':
            output = t.relu(output)
        elif self.non_linear == 'sigmoid':
            output = t.sigmoid(output)
        return output

    def get_graph_infomax_loss(self, node_local_emb, graph_representation, demand_scores, nodes_categories_matrixes):
        '''

        Args:
            node_local_emb: batch_size * n_demand * max_nodes_len * embedding_dim_node
            graph_representation: batch_size * n_demand * embedding_dim_node
            demand_scores: batch_size * n_demand * max_session_len
            nodes_categories_matrixes: batch_size * max_nodes_len* max_session_len
        Returns:
            loss: graph infomax loss, torch.float
        '''
        # Compute nodes weight
        batch_size, n_demand, max_nodes_len, embedding_dim_node = node_local_emb.shape
        node_local_emb = node_local_emb.contiguous() # batch_size * n_demand * max_nodes_len * embedding_dim_node
        nodes_weight = self.get_node_weight(nodes_categories_matrixes,
                                            demand_scores)  # batch_size *n_demand* max_nodes_len * 1
        if self.non_linear_demand_score != 'sigmoid': # 'relu' or None
            nodes_weight = torch.sigmoid(nodes_weight)
        graph_representation = graph_representation.unsqueeze(2).repeat(1, 1, max_nodes_len, 1) # batch_size * n_demand * max_nodes_len * embedding_dim_node
        score = self.f_k(node_local_emb, graph_representation)  # batch_size * n_demand * max_nodes_len * 1

        if nodes_weight.requires_grad:
            nodes_weight.retain_grad()
        target_informax = nodes_weight
        loss = self.loss(score, target_informax.detach())



        return loss


if __name__ == "__main__":
    gnn = GNN(8, 64, 2)
    sess_node_emb = t.randn(32, 10, 64)
    n_demand = 8
    batch_size, max_nodes_len, embedding_dim_node = sess_node_emb.shape
    expand_session_node = sess_node_emb.unsqueeze(1)  # batch_size * 1 * max_nodes_len * embedding_dim_node
    demand_session_node = expand_session_node.expand(batch_size, n_demand, max_nodes_len,
                                                     embedding_dim_node)  # Create n_demnd view of the input node emb
    adj_matrixes = t.randint(0, 2, (32, 10, 10))
    demand_scores = t.randn(32, 8, 15)
    # nodes_categories_matrixes: batch_size * max_nodes_len* max_session_len
    nodes_categories_matrixes = t.randint(0, 2, (32, 10, 15))
    out = demand_session_node
    for l in range(2):
        out = gnn(out, adj_matrixes, demand_scores, nodes_categories_matrixes)
    print(out.shape)
    graph = t.mean(out, dim=2)
    loss = gnn.get_graph_infomax_loss(out, graph, demand_scores, nodes_categories_matrixes)
    print(loss.item())
