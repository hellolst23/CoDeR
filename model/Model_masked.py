# -*- coding: utf-8 -*-
"""
# @File    : Model_masked.py
# Desc:
    mask some parts to test the impact of each component
"""

import torch as t
import torch
from torch import nn
from layer import DemandExtraction, GNN, PVSD


class DemandAwareRS(nn.Module):
    '''
        Proposed DemandAware model
    '''
    def __init__(self, n_items, n_categories, config, gnn_mask=True):
        """
        Args:
            n_items:
            n_categories:
            config:
            gnn_mask: bool, default=False
        """
        super(DemandAwareRS, self).__init__()
        self.gnn_mask = gnn_mask
        self.n_demand = config.n_demand
        self.n_gnn_layer = config.n_gnn_layer
        self.batchNorm = config.batch_norm

        self.demand_extractor = DemandExtraction(config.n_demand, n_categories, config.embedding_dim_c,
                                                 config.hidden_size,config.non_linear_demand_score, config.demand_extract)
        self.embedding_i = nn.Embedding(n_items, config.embedding_dim_i, padding_idx=0)  # itemID starts with 1
        if self.gnn_mask:
            print('!!! Note: no gnn layer component')
        self.gnn_layer = GNN(config.n_demand, config.embedding_dim_i, config.dashed_order, config.bias,
                             config.non_linear, config.non_linear_demand_score, config.demand_share_agg, config.demand_share_node) 
        self.graph_aggregation_method = config.graph_aggregation

        if config.graph_aggregation == 'lstm':
            self.graph_aggregator = nn.LSTM(config.embedding_dim_i, config.embedding_dim_i, batch_first=True) 

        self.p_v_s_d = PVSD(config.embedding_dim_i,config.batch_norm,config.n_demand, config.rs)

    def forward(self, nodes, categories, adj, nodes_categories, session_last_item_index, candidate_category, mask_node):
        '''
        Compute the probability of the candidate item
        Args:
            nodes: torch.Tensor, dtype=torch.int64,  batch_size * max_nodes_len, session unique item sequence.
            categories: torch.Tensor， dtype = torch.int64, batch_size * max_session_len, session items categories sequence.
            adj: torch.Tensor, dtype=torch.float64, batch_size *  max_nodes_len* max_nodes_len, session unique item adjacent matrix.
            nodes_categories: torch.Tensor, dtype=torch.float64, batch_size * max_nodes_len* max_session_len, session items categories to items mapping matrices.
            session_last_item_index: torch.Tensor, dtype=torch.int64, batch_size, record the last item index in unique nodes
            candidate_category: torch.Tensor, dtype=torch.int64, candidate item + 1
            mask_node: torch.Tensor, dtype=torch.int64, batch_size * max_nodes_len, record the clicked nodes in a session
        Returns:
            prediction: torch.Tensor, dytp=torch.float64, batch_size * (n_item + 1), Probability of the all the candidate items.
            infomax loss: torch.Tensor,a scalar,  loss of the graph mutual information
            gnn_node_representation: torch.Tensor, batch_size * n_demand * max_nodes_len * embedding_dim_node
        '''
        device = nodes.device # 'cuda:1' or 'cpu'
        session_emb = self.embedding_i(nodes)
        # Extract demand score
        demand_score, demand_score_candidate = self.demand_extractor(categories, candidate_category)


        # Expand session to demand
        batch_size, max_nodes_len, embedding_dim_node = session_emb.shape
        expand_session_emb = session_emb.unsqueeze(1)  # batch_size * 1 * max_nodes_len * embedding_dim_node
        demand_session_emb = expand_session_emb.expand(batch_size, self.n_demand, max_nodes_len,
                                                       embedding_dim_node)  # Create n_demnd view of the input node emb
        if self.gnn_mask:
            gnn_input = demand_session_emb  # batch_size * n_demand * max_nodes_len * embedding_dim_node
            gnn_node_representation = gnn_input
        else:
            # GNN layer
            gnn_input = demand_session_emb  # batch_size * n_demand * max_nodes_len * embedding_dim_node
            for l in range(self.n_gnn_layer):
                gnn_input = self.gnn_layer(gnn_input, adj, demand_score, nodes_categories)
            gnn_node_representation = gnn_input  # for visualization, batch_size * n_demand * max_nodes_len * embedding_dim_node

        # Aggregate graph
        graph_representation = self.graph_aggregate(gnn_input,mask_node)  # batch_size * n_demand * embedding_dim_node
        infomax_loss = self.gnn_layer.get_graph_infomax_loss(gnn_input, graph_representation, demand_score,
                                                             nodes_categories)

        # Recommendation
        session_last_item_index = session_last_item_index.view(batch_size, 1)
        mask_latest_preference = torch.zeros((batch_size, max_nodes_len), dtype=bool).to(device).scatter(1, session_last_item_index,True)
        mask_latest_preference = mask_latest_preference.view(-1, 1, max_nodes_len, 1)
        last_item_info = gnn_input.masked_select(mask_latest_preference).view(batch_size, self.n_demand,
                                                                              embedding_dim_node)
        # batch_size * n_demand * embedding_dim_node

        candidate_item_emb = self.embedding_i.weight  # (n_item +_1) * embedding_dim_node
        session_representation = t.cat((graph_representation, last_item_info),
                                       dim=2)  # batch_size * n_demand * (embedding_dim_node * 2)

        candidate_item_emb = candidate_item_emb.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.n_demand, 1,
                                                                                 1)  # Batch_size * n_demand * (n_item+1) * embedding_dim_node
        P_v_s_d = self.p_v_s_d(session_representation, candidate_item_emb)  # Batch_size * n_demand * (n_item+1)
        # demand_score_candidate: Batch_size * n_demand * (n_item+1)
        P_v = t.sum(P_v_s_d * demand_score_candidate, dim=1)  # Batch_size  * (n_item+1)

        return P_v, infomax_loss, gnn_node_representation


    def graph_aggregate(self, gnn_result, mask_nodes):
        '''
        Aggregrate graph
        Args:
            gnn_result: item emb after gnn, batch_size * n_demand * max_nodes_len * embedding_dim_node
            mask_nodes: torch.Tensor, dtype=torch.int64, batch_size * max_nodes_len, record the clicked nodes in a session

        Returns:
            graph_representation: batch_size * n_demand * embedding_dim_node
        '''
        batch_size, n_demand, max_nodes_len, embedding_dim_node = gnn_result.shape
        mask_nodes = mask_nodes.float()

        if self.graph_aggregation_method == 'lstm':
            gnn_input_reshape = gnn_result.view(batch_size * n_demand, max_nodes_len,
                                                embedding_dim_node)  # Reshape for parallelization
            _, (graph_representation, _)= self.graph_aggregator(gnn_input_reshape)  # Get the last state, (batch_size * n_demand) * 1 * embedding_dim_node

            graph_representation = graph_representation.view(batch_size, n_demand, embedding_dim_node)
        elif self.graph_aggregation_method == 'mean':
            graph_representation = torch.matmul(mask_nodes.view(batch_size,1,1,max_nodes_len), gnn_result).squeeze(-2)  # batch_size * n_demand * embedding_dim_node
            graph_representation = graph_representation.div(mask_nodes.sum(-1).view(batch_size,1,1))
        elif self.graph_aggregation_method == 'sum':
            graph_representation = torch.matmul(mask_nodes.view(batch_size,1,1,max_nodes_len), gnn_result).squeeze(-2) # batch_size * n_demand * embedding_dim_node
        return graph_representation

if __name__ == '__main__':
    """
    # Model config
    n_demand = 2  # Demand number
    embedding_dim_i = 10  # emb dim for item
    embedding_dim_c = 10  # emb dim for categories
    hidden_size = 15  # hidden size for demand extraction
    dashed_order = 2  # order of dashed edges
    bias = False
    non_linear = 'relu'
    n_gnn_layer = 2  # Number of the gnn layer
    graph_aggregation = 'lstm'  # method to aggregate the graph, lstm, mean, sum...
    """
    """
        nodes: torch.Tensor, dtype=torch.int64,  batch_size * max_nodes_len, session unique item sequence.
                categories: torch.Tensor， dtype = torch.int64, batch_size * max_session_len, session items categories sequence.
                adj: torch.Tensor, dtype=torch.float64, batch_size *  max_nodes_len* max_nodes_len, session unique item adjacent matrix.
                nodes_categories: torch.Tensor, dtype=torch.float64, batch_size * max_nodes_len* max_session_len, session items categories to items mapping matrices.
                session_last_item_index: torch.Tensor, dtype=torch.int64, batch_size, Index of the the last item in session in sess_nodes_batch.
                candidate_category: torch.Tensor, dtype=torch.int64, candidate item + 1
        """
    # Test model
    """
    import test_config as config

    model = DemandAwareRS(n_items=32, n_categories=36, config=config)

    sess_nodes = t.LongTensor(t.arange(1, 31).view(5, 6))  # batch_size=5, max_nodes_len=6
    sess_categories = t.LongTensor(t.arange(1, 36, ).view(5, 7))  # max_session_len=7
    adj_matrixes = t.FloatTensor(t.ones(5, 6, 6))
    adj_matrixes[:, 3:, 3:] = 0
    nodes_categories_matrixes = t.FloatTensor(t.ones(5, 6, 7))  # batch_size=5 * max_nodes_len=6 * max_session_len=7
    session_last_item_index = t.LongTensor(t.arange(1, 6))
    candidate_category = t.LongTensor(t.arange(0, 32))# n_items=32，
    mask_node = t.LongTensor(t.ones((5,6), dtype=int))
    mask_node[:,4:] = 0

    score,informax_loss, _ = model(sess_nodes, sess_categories, adj_matrixes, nodes_categories_matrixes, session_last_item_index, candidate_category, mask_node)
    print(informax_loss.shape)
    """
