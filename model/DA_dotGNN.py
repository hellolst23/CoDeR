
# -*- coding: utf-8 -*-
"""
# @File    : DA_dotGNN.py
"""
import math
import torch as t
import torch
from torch import nn
from layer import DemandExtraction, MeanGNN, PVSD
from util.utils import file_write

print(" -----------import DA_dotGNN.py------------")


class DemandAwareRS(nn.Module):
    '''
    Proposed DemandAware model
    '''
    def __init__(self, n_items, n_categories, config):
        """
        Args:
            n_items:
            n_categories:
            config:
        """

        super(DemandAwareRS, self).__init__()
        self.recommend_model = config.recommend_model
        self.softmax_demand_score_candidate = config.softmax_demand_score_candidate

        self.gnn_mask = config.gnn_mask
        self.n_demand = config.n_demand
        self.n_gnn_layer = config.n_gnn_layer
        self.nonhybrid = config.nonhybrid
        self.batchNorm = config.batch_norm
        self.embedding_dim_node = config.embedding_dim_i
        self.hidden_size = config.hidden_size
        self.log_path_txt = config.log_path_txt

        self.embed_l2 = config.embed_l2

        if config.demand_mask == False:
            file_write(self.log_path_txt, 'There is DemandExtraction part')
        else:
            file_write(self.log_path_txt, '!!! Note: no demand extraction part, weight is equal for each node and demand')

        self.demand_extractor = DemandExtraction(config.n_demand, n_categories, config.embedding_dim_c,
                                                 config.hidden_size,config.non_linear_demand_score, config.demand_extract, demand_mask=config.demand_mask, demand_agg=config.demand_agg)
        self.embedding_i = nn.Embedding(n_items, config.embedding_dim_i, padding_idx=0)  # itemID starts with 1

        if self.gnn_mask:
            file_write(self.log_path_txt, '!!! Note: no gnn layer component')
        else:
            file_write(self.log_path_txt, 'There is GNN part')

        self.gnn_layer = MeanGNN(config,config.n_demand, config.embedding_dim_i, config.dashed_order, config.bias,
                             config.non_linear, config.non_linear_demand_score, config.demand_share_agg, config.demand_share_node) 
        self.graph_aggregation_method = config.graph_aggregation

        if config.graph_aggregation == 'lstm':
            self.graph_aggregator = nn.LSTM(config.embedding_dim_i, config.embedding_dim_i, batch_first=True) 

        self.p_v_s_d = PVSD(config.embedding_dim_i,config.batch_norm,config.n_demand, config.rs)

        if self.recommend_model == 'sr_gnn':
            file_write(self.log_path_txt, f'recommend_model: {self.recommend_model}')
            self.linear_one = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            self.linear_two = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            self.linear_three = nn.Linear(config.hidden_size, 1, bias=False)
            self.linear_transform = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=True)
        else:
            file_write(self.log_path_txt, 'recommend_model: DemandRS')
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        stdv_i = 1.0 / math.sqrt(self.embedding_dim_node)
        for name_parameter, weight in self.named_parameters():
            #print(name_parameter, ': ', weight)
            if name_parameter == 'embedding_i.weight':
                weight.data.uniform_(-stdv_i, stdv_i)
            else:
                weight.data.uniform_(-stdv, stdv)

    def recomend_srgnn(self, hidden, demand_score_candidate, session_last_item_index, mask_node, batch_size):
        """
    
        Args:
            hidden: torch.Tensor, dtype=torch.int64, batch_size * n_demand * max_nodes_len * embedding_dim_node, node representation after gnn layer
            demand_score_candidate: torch.Tensor, dtype=torch.float32  batch_size * n_demand * n_items
            session_last_item_index:
            mask_node: torch.Tensor, dtype=torch.int64, batch_size * max_nodes_len, record the clicked nodes in a session
            batch_size:

        Returns:

        """
        max_nodes_len = hidden.shape[2]
        #hidden: batch_size * n_demand * max_nodes_len * embedding_dim_node,
        last_item_info = hidden.transpose(2, 1)[torch.arange(
            len(session_last_item_index)), session_last_item_index]  # batch_size * n_demand * embedding_dim_node
  
        q1 = self.linear_one(last_item_info).unsqueeze(2) # batch_size * n_demand * 1 * embedding_dim_node
        q2 = self.linear_two(hidden) # batch_size * n_demand * max_nodes_len * embedding_dim_node
        alpha = self.linear_three(torch.sigmoid(q1 + q2)) # batch_size * n_demand * max_nodes_len * 1

        hidden = alpha * hidden #  batch_size * n_demand * max_nodes_len * embedding_dim_node


        a = torch.matmul(mask_node.view(batch_size,1,1,max_nodes_len).float(), hidden).squeeze(-2) # batch_size * n_demand * embedding_dim_node
        graph_representation = a

        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, last_item_info], dim=-1))
   

        # todo ################################################################
        b = self.embedding_i.weight.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.n_demand, 1,1)
        # include 0, Batch_size * n_demand * (n_item) * embedding_dim_node

        P_v_s_d = torch.matmul(b, a.unsqueeze(-1)).squeeze(-1) # batch_size * n_demand * (n_item)
        #print('P_v_s_d: ', P_v_s_d.shape)
        #print('demand_score_candidate: ', demand_score_candidate.shape)
        P_v = t.sum(P_v_s_d * demand_score_candidate, dim=1)  # Batch_size  * (n_item+1)
        # batch_size * n_demand * n_items
        return P_v, graph_representation


    def recommend_demand(self, hidden, demand_score_candidate, session_last_item_index, mask_node, batch_size):
        """
        
        Args:
            hidden: torch.Tensor, dtype=torch.int64, batch_size * n_demand * max_nodes_len * embedding_dim_node, node representation after gnn layer
            demand_score_candidate: torch.Tensor, dtype=torch.float32  batch_size * n_demand * n_items
            session_last_item_index: torch.Tensor, dtype=torch.int64, batch_size, record the last item index in unique nodes
            mask_node: torch.Tensor, dtype=torch.int64, batch_size * max_nodes_len, record the clicked nodes in a session
            batch_size:

        Returns:
        P_v: torch.Tensor, dyte=torch.float64, batch_size * (n_item + 1), Probability of the all the candidate items.
        graph_representation: torch.Tensor, dtype=torch.float64, batch_size * n_demand * embedding_dim_node
        """

        # Aggregate graph
        graph_representation = self.graph_aggregate(hidden, mask_node)  # batch_size * n_demand * embedding_dim_node

        # Recommendation
        last_item_info = hidden.transpose(2, 1)[torch.arange(
            len(session_last_item_index)), session_last_item_index]  # batch_size * n_demand * embedding_dim_node
        session_representation = t.cat((graph_representation, last_item_info),
                                       dim=2)  # batch_size * n_demand * (embedding_dim_node * 2)
        P_v = self.compute_score(session_representation, demand_score_candidate, batch_size)
        return P_v, graph_representation

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
        demand_score, demand_score_candidate, _, __ = self.demand_extractor(categories, candidate_category)
        # demand_score:torch.Tensor, dtype=torch.float32  batch_size * n_demand * max_session_len
        # demand_score_candidate:  torch.Tensor, dtype=torch.float32  batch_size * n_demand * n_items

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


        l2_loss = self.regularize(gnn_input, mask_node)

        if self.recommend_model == 'sr_gnn':
            P_v, graph_representation = self.recomend_srgnn(gnn_input, demand_score_candidate,session_last_item_index, mask_node, batch_size)
        else:
            P_v, graph_representation = self.recommend_demand(gnn_input, demand_score_candidate,session_last_item_index, mask_node, batch_size)
        infomax_loss = self.gnn_layer.get_graph_infomax_loss(gnn_input, graph_representation, demand_score, nodes_categories)

        return P_v, infomax_loss,l2_loss, gnn_node_representation



    def compute_score(self, session_representation,demand_score_candidate, batch_size): 
        """
        compute P_v
        Args:
            session_representation: torch.Tensor, dtype=torch.float64, batch_size * n_demand * (embedding_dim_node * 2)
            demand_score_candidate: torch.Tensor, dtype=torch.float32  batch_size * n_demand * n_items
            batch_size:
        Returns:
        P_v: torch.Tensor, dytp=torch.float64, batch_size * (n_item + 1), Probability of the all the candidate items.
        """

        candidate_item_emb = self.embedding_i.weight # (n_item +_1) * embedding_dim_node
        candidate_item_emb = candidate_item_emb.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.n_demand, 1,
                                                                                 1)  # Batch_size * n_demand * (n_item+1) * embedding_dim_node

        P_v_s_d = self.p_v_s_d(session_representation, candidate_item_emb)  # Batch_size * n_demand * (n_item+1)
        # demand_score_candidate: Batch_size * n_demand * (n_item+1)
        if self.softmax_demand_score_candidate:
            demand_score_candidate = torch.softmax(demand_score_candidate, dim=1)
        P_v = t.sum(P_v_s_d * demand_score_candidate, dim=1)  # Batch_size  * (n_item+1)

        return P_v

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

    def regularize(self, gnn_result, mask_nodes):
        """
        Args:
            gnn_result: item emb after gnn, batch_size * n_demand * max_nodes_len * embedding_dim_node
            mask_nodes: torch.Tensor, dtype=torch.int64, batch_size * max_nodes_len, record the clicked nodes in a session
        Returns:
        """
        batch_size, n_demand, max_nodes_len, embedding_dim_node = gnn_result.shape
        mask_nodes = mask_nodes.float()

        temp = torch.matmul(mask_nodes.view(batch_size, 1, 1, max_nodes_len), (gnn_result ** 2)).squeeze(
            -2)  # batch_size * n_demand * embedding_dim_node
        l2_loss = self.embed_l2 * temp.sum()
        return l2_loss
