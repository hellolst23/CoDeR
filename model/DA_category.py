# -*- coding: utf-8 -*-
"""
# Desc:
add category embedding while recommendation and with MeanGNN
"""
import math
import torch as t
import torch
from torch import nn
from util.utils import file_write

from layer import DemandExtraction, PVSD, MeanGNN
#
# from layer import PVSD, MeanGNN
# from layer.DemandExtraction_test import DemandExtraction

print(" -----------import DA_category.py------------")


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
        # 下面注释 n_item + 1 = n_items
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

        self.catgy_embedding = config.catgy_embedding

        self.dropout = nn.Dropout(config.drop_item)
        self.add_pos = config.add_pos
        if self.add_pos:
            file_write(self.log_path_txt, 'Add pos embedding')
            self.pos_embedding = nn.Embedding(50, config.embedding_dim_i)  # NOTE: add pos embedding
            self.w_pos = nn.Linear(2 * config.embedding_dim_i, config.embedding_dim_i)
        if self.catgy_embedding:
            self.dim_coefficient = 2
            file_write(self.log_path_txt, 'Catgy embedding is passed into GCN part')
        else:
            self.dim_coefficient = 1

        if config.demand_mask == False:
            file_write(self.log_path_txt, 'There is DemandExtraction part')
        else:
            file_write(self.log_path_txt,
                           'Note: demand extraction part, weight is equal for each node and demand')

        self.demand_extractor = DemandExtraction(config.n_demand, n_categories, config.embedding_dim_c,
                                                 config.hidden_size, config.drop_catgy,config.predict_catgy_fun, config.non_linear_demand_score,
                                                 config.demand_extract,demand_mask=config.demand_mask,
                                                 demand_agg=config.demand_agg)
        self.embedding_i = nn.Embedding(n_items, config.embedding_dim_i, padding_idx=0)  # itemID starts with 1

        if self.gnn_mask:
            file_write(self.log_path_txt, '!!! Note: no gnn layer component')
        else:
            file_write(self.log_path_txt, 'There is GNN part')
        self.gnn_layer = MeanGNN(config, config.n_demand, self.dim_coefficient * config.embedding_dim_i, config.dashed_order, config.bias,
                                 config.non_linear, config.non_linear_demand_score, config.demand_share_agg,
                                 config.demand_share_node)  # TODO: How many layer to use?
        self.graph_aggregation_method = config.graph_aggregation

        if config.graph_aggregation == 'lstm':
            self.graph_aggregator = nn.LSTM(config.embedding_dim_i, config.embedding_dim_i,
                                            batch_first=True)  # todo 不知道后面补的0对其是否有影响

        self.p_v_s_d = PVSD(2 * config.embedding_dim_i, config.batch_norm, config.n_demand, config.rs)

        if self.recommend_model == 'sr_gnn':

            file_write(self.log_path_txt, f'recommend_model: {self.recommend_model}')
            self.linear_one = nn.Linear(self.dim_coefficient * config.hidden_size, config.hidden_size, bias=True)
            self.linear_two = nn.Linear(self.dim_coefficient * config.hidden_size, config.hidden_size, bias=True)
            self.linear_three = nn.Linear(config.hidden_size, 1, bias=False)
            self.linear_transform = nn.Linear(config.hidden_size * self.dim_coefficient * 2, config.hidden_size * self.dim_coefficient, bias=True)
        else:
            file_write(self.log_path_txt, 'recommend_model: DemandRS')


        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        stdv_i = 1.0 / math.sqrt(self.embedding_dim_node)
        for name_parameter, weight in self.named_parameters():
            # print(name_parameter, ': ', weight)
            if name_parameter == 'embedding_i.weight':
                weight.data.uniform_(-stdv_i, stdv_i)
            else:
                weight.data.uniform_(-stdv, stdv)

    def recomend_srgnn(self, hidden, demand_score_candidate, category_candidadte, session_last_item_index, mask_node,
                       batch_size):
        """
        This recommend past is same to SR-GNN，https://arxiv.org/abs/1811.00855
        Args:
            hidden: torch.Tensor, dtype=torch.int64, batch_size * n_demand * max_nodes_len * embedding_dim_node, node representation after gnn layer
            demand_score_candidate: torch.Tensor, dtype=torch.float32  batch_size * n_demand * n_items
            category_candidadte: torch.Tensor, dtype=torch.float32, n_items * embedding_dim_c
            session_last_item_index:
            mask_node: torch.Tensor, dtype=torch.int64, batch_size * max_nodes_len, record the clicked nodes in a session
            batch_size:

        Returns:

        """
        max_nodes_len = hidden.shape[2]
        # hidden: batch_size * n_demand * max_nodes_len * embedding_dim_node,
        last_item_info = hidden.transpose(2, 1)[torch.arange(
            len(session_last_item_index)), session_last_item_index]  # batch_size * n_demand * embedding_dim_node
        # todo 这里不同的demand 对应的全连接参数共享

        if self.gnn_mask:
            a = torch.matmul(mask_node.view(batch_size, 1, 1, max_nodes_len).float(), hidden).squeeze(
                -2)  # batch_size * n_demand * embedding_dim_node
            a = a.div(mask_node.sum(1).unsqueeze(-1).unsqueeze(-1))  # NOTE: 取均值
            graph_representation = a
        else:
            q1 = self.linear_one(last_item_info).unsqueeze(2)  # batch_size * n_demand * 1 * embedding_dim_node
            q2 = self.linear_two(hidden)  # batch_size * n_demand * max_nodes_len * embedding_dim_node
            alpha = self.linear_three(torch.sigmoid(q1 + q2))  # batch_size * n_demand * max_nodes_len * 1

            hidden = alpha * hidden  # batch_size * n_demand * max_nodes_len * embedding_dim_node
            a = torch.matmul(mask_node.view(batch_size, 1, 1, max_nodes_len).float(), hidden).squeeze(
                -2)  # batch_size * n_demand * embedding_dim_node
            graph_representation = a

        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, last_item_info], dim=-1))
        # b_i = self.embedding.weight[1:]  # n_nodes x latent_size，这里把数据集合中所有的item都当成candidate sets
        b = self.embedding_i.weight
        if self.catgy_embedding:
            b = torch.cat((b, category_candidadte), dim=-1)
            # b = self.linear_transform(b)
        b = b.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.n_demand, 1, 1)
        # include 0, Batch_size * n_demand * (n_item) * embedding_dim_node

        P_v_s_d = torch.matmul(b, a.unsqueeze(-1)).squeeze(-1)  # batch_size * n_demand * (n_item)
        self.p_score = P_v_s_d
        # print('P_v_s_d: ', P_v_s_d.shape)
        # print('demand_score_candidate: ', .shape)
        # if self.p_v_compute = config.p_v_computedemand_score_candidate
        P_v = t.sum(P_v_s_d * demand_score_candidate, dim=1)  # Batch_size  * (n_item+1) # todo
        # batch_size * n_demand * n_items
        # P_v = t.sum(P_v_s_d +  demand_score_candidate, dim=1) # NOTE： 改成了相加
        return P_v, graph_representation

    def recommend_demand(self, hidden, demand_score_candidate, category_candidadte, session_last_item_index, mask_node,
                         batch_size):
        """
        This recommend part 按照自己写的公式
        Args:
            hidden: torch.Tensor, dtype=torch.int64, batch_size * n_demand * max_nodes_len * embedding_dim_node, node representation after gnn layer
            demand_score_candidate: torch.Tensor, dtype=torch.float32  batch_size * n_demand * n_items
            category_candidadte: torch.Tensor, dtype=torch.float32, n_items * embedding_dim_c
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
        P_v = self.compute_score(session_representation, demand_score_candidate, category_candidadte, batch_size)
        return P_v, graph_representation

    def get_demand_Score(self):
        '''
        Return the demand score and PVSD
        Returns:
            self.demand_score: torch.Tensor, dtype=torch.float32  batch_size * n_demand * max_session_len
            self.P_v_s_d: # batch_size * n_demand * (n_item)
        '''
        return self.demand_score, self.p_score
    def forward(self, nodes, categories, adj, nodes_categories, session_last_item_index, candidate_category, mask_node,session_last_catgy_index, mask_catgy):
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
            session_last_catgy_index_batch: torch.Tensor, dtype=torch.int64, batch_size, record the last item index in catgy session
            mask_catgy_batch: torch.Tensor, dtype=torch.int64, batch_size * max_session_len, record the clicked catgy in a session

        Returns:
            prediction: torch.Tensor, dytp=torch.float64, batch_size * (n_item + 1), Probability of the all the candidate items.
            infomax loss: torch.Tensor,a scalar,  loss of the graph mutual information
            gnn_node_representation: torch.Tensor, batch_size * n_demand * max_nodes_len * embedding_dim_node
            demand_sim_loss: demand similarity loss torch.float64
        '''
        device = nodes.device  # 'cuda:1' or 'cpu'
        session_emb = self.embedding_i(nodes)  # batch_size * max_nodes_len * embedding_dim_node
        session_emb = self.dropout(session_emb) # NOTE: add drop out

        batch_size, max_nodes_len, embedding_dim_node = session_emb.shape
        if self.add_pos:
            session_emb_pos = self.pos_embedding.weight[:max_nodes_len].unsqueeze(0).repeat(batch_size, 1,
                                                                                            1)  # 1 * max_nodes_len * embedding_dim_node
            # session_emb = self.w_pos(torch.cat([session_emb, session_emb_pos], dim=-1))
            session_emb = session_emb + session_emb_pos
            session_emb = torch.tanh(session_emb)

        # Extract demand score
        # demand_score, demand_score_candidate, emdedding_c, embedding_c_candidadte, demand_sim_loss = self.demand_extractor(
        #     categories, candidate_category)
        catgy_score, demand_score, demand_score_candidate, emdedding_c, embedding_c_candidadte, demand_sim_loss = self.demand_extractor(
            categories, candidate_category, session_last_catgy_index, mask_catgy)
        # demand_score:torch.Tensor, dtype=torch.float32  batch_size * n_demand * max_session_len
        # demand_score_candidate:  torch.Tensor, dtype=torch.float32  batch_size * n_demand * n_items
        # emdedding_c:torch.Tensor, dtype=torch.float32  batch_size * max_session_len * embedding_dim_c
        # embedding_c_candidadte: torch.Tensor, dtype=torch.float32 n_items * embedding_dim_c
        self.demand_score = demand_score
        # Expand session to demand
        # batch_size, max_nodes_len, embedding_dim_node = session_emb.shape
        if self.catgy_embedding:
            temp_node_catgy = self.normalize_matrix(nodes_categories)
            emdedding_c = torch.matmul(temp_node_catgy, emdedding_c)  # batch_size * max_nodes_len * embedding_dim_c
            # 　session_emb = emdedding_c + session_emb
            session_emb = torch.cat((session_emb, emdedding_c), dim=-1)

        expand_session_emb = session_emb.unsqueeze(1)  # batch_size * 1 * max_nodes_len * (2 embedding_dim_node)
        demand_session_emb = expand_session_emb.expand(batch_size, self.n_demand, max_nodes_len,
                                                       self.dim_coefficient * embedding_dim_node)  # Create n_demnd view of the input node emb
        if self.gnn_mask:
            gnn_input = demand_session_emb  # batch_size * n_demand * max_nodes_len *(2 embedding_dim_node)
            gnn_node_representation = gnn_input
        else:
            # GNN layer
            gnn_input = demand_session_emb  # batch_size * n_demand * max_nodes_len * (2 embedding_dim_node)
            for l in range(self.n_gnn_layer):
                gnn_input = self.gnn_layer(gnn_input, adj, demand_score, nodes_categories)
            gnn_node_representation = gnn_input  # for visualization, batch_size * n_demand * max_nodes_len * (2 embedding_dim_node)

        # 计算 L2 loss
        l2_loss = self.regularize(gnn_input, mask_node)

        if self.recommend_model == 'sr_gnn':
            P_v, graph_representation = self.recomend_srgnn(gnn_input, demand_score_candidate, embedding_c_candidadte,
                                                            session_last_item_index, mask_node, batch_size)
        else:
            P_v, graph_representation = self.recommend_demand(gnn_input, demand_score_candidate, embedding_c_candidadte,
                                                              session_last_item_index, mask_node, batch_size)
        infomax_loss = self.gnn_layer.get_graph_infomax_loss(gnn_input, graph_representation, demand_score,
                                                             nodes_categories)

        return P_v, catgy_score,infomax_loss, l2_loss, gnn_node_representation, demand_sim_loss

    def normalize_matrix(self, matrix):
        """
        a_ij = a_ij / sum({a_ik: k \in [1,...,n}), Normalize matrix by row
        Args:
            matrix: batch_size * max_nodes_len * max_session_len,   #target * source
        Returns:
        """
        rowsum = matrix.sum(2).clamp(1, matrix.shape[-1])
        matrix = matrix / rowsum.unsqueeze(-1).float()
        return matrix

    def compute_score(self, session_representation, demand_score_candidate, category_candidadte,
                      batch_size):
        """
        compute P_v
        Args:
            session_representation: torch.Tensor, dtype=torch.float64, batch_size * n_demand * (embedding_dim_node * 2)
            demand_score_candidate: torch.Tensor, dtype=torch.float32  batch_size * n_demand * n_items
            category_candidadte: torch.Tensor, dtype=torch.float32, n_items * embedding_dim_c
            batch_size:
        Returns:
        P_v: torch.Tensor, dytp=torch.float64, batch_size * (n_item + 1), Probability of the all the candidate items.
        """

        candidate_item_emb = self.embedding_i.weight  # (n_item +_1) * embedding_dim_node
        if self.catgy_embedding:
            # candidate_item_emb = candidate_item_emb + category_candidadte
            candidate_item_emb = torch.cat((candidate_item_emb, category_candidadte),
                                           dim=-1)  # (n_item +_1) * （2 embedding_dim_node）
            # candidate_item_emb = self.fc_cat1(candidate_item_emb)
        candidate_item_emb = candidate_item_emb.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.n_demand, 1,
                                                                                 1)  # Batch_size * n_demand * (n_item+1) * （2 embedding_dim_node）

        P_v_s_d = self.p_v_s_d(session_representation, candidate_item_emb)  # Batch_size * n_demand * (n_item+1)
        self.p_score = P_v_s_d
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
            _, (graph_representation, _) = self.graph_aggregator(
                gnn_input_reshape)  # Get the last state, (batch_size * n_demand) * 1 * embedding_dim_node
            # fixme 我测出来是 1 * (batch_size * n_demand) * embedding_dim_node
            graph_representation = graph_representation.view(batch_size, n_demand, embedding_dim_node)
        elif self.graph_aggregation_method == 'mean':
            graph_representation = torch.matmul(mask_nodes.view(batch_size, 1, 1, max_nodes_len), gnn_result).squeeze(
                -2)  # batch_size * n_demand * embedding_dim_node
            graph_representation = graph_representation.div(mask_nodes.sum(-1).view(batch_size, 1, 1))
        elif self.graph_aggregation_method == 'sum':
            graph_representation = torch.matmul(mask_nodes.view(batch_size, 1, 1, max_nodes_len), gnn_result).squeeze(
                -2)  # batch_size * n_demand * embedding_dim_node
        return graph_representation

    def regularize(self, gnn_result, mask_nodes):
        """
        References: BGCN , 对最终预测前的embedding做 L2 正则,
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
    import test_config as config

    model = DemandAwareRS(n_items=32, n_categories=36, config=config)  # n_items: 算补的0

    sess_nodes = t.LongTensor(torch.arange(1, 31).view(5, 6))  # batch_size=5, max_nodes_len=6
    sess_categories = t.LongTensor(torch.arange(1, 36, ).view(5, 7))  # max_session_len=7
    adj_matrixes = t.FloatTensor(torch.ones(5, 6, 6))
    adj_matrixes[:, 3:, 3:] = 0
    nodes_categories_matrixes = t.FloatTensor(t.ones(5, 6, 7))  # batch_size=5 * max_nodes_len=6 * max_session_len=7
    session_last_item_index = t.LongTensor(t.arange(1, 6))
    candidate_category = t.LongTensor(t.arange(0, 32))  # n_items=32，
    mask_node = t.LongTensor(t.ones((5, 6), dtype=int))
    mask_node[:, 4:] = 0

    score, informax_loss, _ = model(sess_nodes, sess_categories, adj_matrixes, nodes_categories_matrixes,
                                    session_last_item_index, candidate_category, mask_node)
    print(informax_loss.shape)