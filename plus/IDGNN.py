import torch
import torch as t
from torch import nn
from torch.nn import Module
from layer.Linear3D import Linear3D


class IDemandGNN(Module):
    def __init__(self, config, n_demand, embedding_dim_node, bias=True):
        super(IDemandGNN, self).__init__()
        '''
        Args:
            config: configurations
            n_demand: number of the demand
            embedding_dim_node: embedding dimension of the node
        '''
        self.n_demand = n_demand
        self.plus = config.plus
        self.gnn_weight = Linear3D(self.n_demand, 3*embedding_dim_node, embedding_dim_node, bias=bias)

    def get_node_weight(self, nodes_categories_matrixes, demand_scores):
        """
        Args:
            nodes_categories_matrixes: batch_size * max_nodes_len* max_session_len
            demand_scores: torch.Tensor, batch_size * n_demand * max_session_len
        Returns:
            nodes_weight: torch.Tensor, batch_size * n_demand * max_nodes_len * 1
        """
        nodes_categories_matrixes = self.normalize_matrix(nodes_categories_matrixes)
        nodes_categories_matrixes = nodes_categories_matrixes.unsqueeze(1)
        demand_scores = demand_scores.unsqueeze(-1)
        nodes_weight = nodes_categories_matrixes.matmul(demand_scores)

        return nodes_weight

    def get_edge_weight(self, nodes_categories_matrixes, interest_ratings):
        """
        Args:
            nodes_categories_matrixes: batch_size * max_nodes_len * max_session_len
            interest_ratings: torch.Tensor, batch_size * n_interest * max_session_len
        Returns:
            edges_weight: torch.tensor, batch_size * max_nodes_len * 1
        """
        nodes_categories_matrixes = self.normalize_matrix(nodes_categories_matrixes)
        nodes_categories_matrixes = nodes_categories_matrixes.unsqueeze(1)
        interest_ratings = interest_ratings.unsqueeze(-1)
        edges_weight = nodes_categories_matrixes.matmul(interest_ratings)
        edges_weight = edges_weight.mean(1)

        return edges_weight
    
    def normalize_matrix(self, matrix):
        """
        Args:
            matrix: batch_size * max_nodes_len * max_session_len
        Returns:
            matrix: normalized matrix
        """
        rowsum = matrix.sum(2).clamp(1, matrix.shape[-1])
        matrix = matrix / rowsum.unsqueeze(-1).float()
        return matrix
    
    def forward(self, demand_session_node, adj_matrixes, demand_scores, interst_rating, nodes_categories_matrixes):
        """
        Args:
            demand_session_node: batch_size * n_demand * max_nodes_len * embedding_dim_node
            adj_matrixes: adjacent matrix, batch_size * max_nodes_len * max_nodes_len
            demand_scores: batch_size * n_demand * max_session_len
            interst_rating: torch.Tensor, dtype=torch.float32, batch_size * n_interest * max_session_len
            nodes_categories_matrixes: batch_size * max_nodes_len* max_session_len
        Returns:
            output: batch_size * n_demand * max_nodes_len * embedding_dim_node
        """
        nodes_weight = self.get_node_weight(nodes_categories_matrixes, demand_scores)
        edge_weight = self.get_edge_weight(nodes_categories_matrixes, interst_rating)
        weight_demand_emb = nodes_weight * demand_session_node
        order_emb_output = []
        weighted_adj = edge_weight * adj_matrixes
        weighted_adj = weighted_adj.permute(0, 2, 1).float()
        adj_matrixes = adj_matrixes.permute(0, 2, 1).float()
        k_order_matrix = adj_matrixes
        normalize_k_matrix_in = self.normalize_matrix(k_order_matrix)
        gnn_output = t.matmul(normalize_k_matrix_in.unsqueeze(1),weight_demand_emb)
        order_emb_output.append(gnn_output)
        k_order_matrix = (k_order_matrix @ weighted_adj).clamp(0, 1)
        normalize_k_matrix = self.normalize_matrix(k_order_matrix)
        k_demand_matrix = []
        for j in range(self.n_demand):
            dashed_edges_weight = t.log(t.matmul(nodes_weight[:, j, :, :],
                                                 nodes_weight[:, j, :, :].exp().permute(0, 2, 1)) + 1)
            demand_matrix = dashed_edges_weight * normalize_k_matrix
            k_demand_matrix.append(demand_matrix)
        k_demand_matrix = t.stack(k_demand_matrix, dim=1)
        gnn_output = t.matmul(k_demand_matrix, weight_demand_emb)
        order_emb_output.append(gnn_output)
        order_emb_output = torch.cat(order_emb_output, dim=-1)
        output = self.gnn_weight(torch.cat((demand_session_node, order_emb_output), dim=-1))
        output = t.relu(output)
        
        return output