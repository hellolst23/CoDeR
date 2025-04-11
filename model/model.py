import os
import math
import torch
import torch as t
import numpy as np
from torch import nn
from layer.match import Match
from plus.IDGNN import IDemandGNN
from plus.IDExtraction import IDExtraction
from layer.position import PositionalEmbedding
from layer.KL import get_kl_score
from layer.modularity import get_modularity, modularity

class CoDeR(nn.Module):
    def __init__(self, n_items, n_categories, config):
        """
        Args:
            n_items: number of items
            n_categories: number of categories
            config: configurations
        """
        super(CoDeR, self).__init__()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_demand = config.n_demand
        self.plus = config.plus
        self.n_interest = config.n_interest
        self.n_gnn_layer = config.n_gnn_layer
        self.embedding_dim_node = config.embedding_dim_i
        self.hidden_size = config.hidden_size
        self.embed_l2 = config.embed_l2
        self.alpha = config.alpha
        self.ad = config.ad

        self.dropout = nn.Dropout(config.drop_item)
        if not self.ad:
            self.pos_embedding = nn.Embedding(50, config.embedding_dim_i)
            self.embedding_position = PositionalEmbedding(50, config.embedding_dim_i)
            self.w_pos = nn.Linear(2 * config.embedding_dim_i, config.embedding_dim_i)
        self.i_demand_extractor = IDExtraction(self.n_interest, config.n_demand, n_categories, config.embedding_dim_c,
                                                 config.hidden_size, config.drop_catgy,config.predict_catgy_fun)
        self.embedding_i = nn.Embedding(n_items, config.embedding_dim_i, padding_idx=0)
        self.gnn_layer = IDemandGNN(config, config.n_demand, 2 * config.embedding_dim_i, config.bias)
        self.match = Match(2 * config.embedding_dim_i, config.batch_norm, config.n_demand, config.rs)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        stdv_i = 1.0 / math.sqrt(self.embedding_dim_node)
        for name_parameter, weight in self.named_parameters():
            if name_parameter == 'embedding_i.weight':
                weight.data.uniform_(-stdv_i, stdv_i)
            else:
                weight.data.uniform_(-stdv, stdv)


    def recommend_demand(self, hidden, demand_score_candidate, category_candidadte, sess_nodes, session_last_item, mask_node,
                         item_category, adj_matrixes, sess_categories, batch_size):
        """
        Args:
            hidden: torch.Tensor, dtype=torch.int64, batch_size * n_demand * max_nodes_len * embedding_dim_node
            demand_score_candidate: torch.Tensor, dtype=torch.float32  batch_size * n_demand * n_items
            category_candidadte: torch.Tensor, dtype=torch.float32, n_items * embedding_dim_c
            sess_nodes: torch.Tensor, dtype=torch.int64,  batch_size * max_nodes_len
            session_last_item: torch.Tensor, dtype=torch.int64, batch_size
            mask_node: torch.Tensor, dtype=torch.int64, batch_size * max_nodes_len
            item_category: a dic, dtype={int: list[int]}
            adj_matrixes: torch.Tensor, dtype=torch.float64, batch_size *  max_nodes_len * max_nodes_len
            sess_categories: torch.Tensor, dtype = torch.int64, batch_size * max_session_len
            batch_size: batch_size
        Returns:
            result_click: torch.Tensor, dyte=torch.float64, batch_size * (n_item + 1)
        """

        _, _, max_nodes_len, _ = hidden.shape
        mask_nodes = mask_node.float()
        graph_representation = torch.matmul(mask_nodes.view(batch_size, 1, 1, max_nodes_len), hidden).squeeze(-2)
        graph_representation = graph_representation.div(mask_nodes.sum(-1).view(batch_size, 1, 1))

        batch_indices = np.arange(len(session_last_item))
        item_last = sess_nodes[batch_indices, session_last_item]
        if self.ad:
            session_last_item = self.deconfounder_esti(sess_nodes.cpu().detach().numpy(),
                                                   session_last_item.cpu().detach().numpy(), item_last.cpu().detach().numpy(), item_category,
                                                   adj_matrixes.cpu().detach().numpy(), sess_categories.cpu().detach().numpy())
        last_item_info = hidden.transpose(2, 1)[torch.arange(len(session_last_item)), session_last_item]
        session_representation = t.cat((graph_representation, last_item_info), dim=2)

        result_click = self.compute_score(session_representation, demand_score_candidate, category_candidadte, batch_size)
        return result_click
    
    def compute_score(self, session_representation, demand_score_candidate, category_candidadte, batch_size):
        """
        Args:
            session_representation: torch.Tensor, dtype=torch.float64, batch_size * n_demand * (embedding_dim_node * 2)
            demand_score_candidate: torch.Tensor, dtype=torch.float32  batch_size * n_demand * n_items
            category_candidadte: torch.Tensor, dtype=torch.float32, n_items * embedding_dim_c
            batch_size: batch_size
        Returns:
            result_click: torch.Tensor, dytp=torch.float64, batch_size * (n_item + 1)
        """
        candidate_item_emb = self.embedding_i.weight
        candidate_item_emb = torch.cat((candidate_item_emb, category_candidadte),dim=-1)
        candidate_item_emb = candidate_item_emb.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.n_demand, 1, 1)

        p_score = self.match(session_representation, candidate_item_emb)
        self.p_score = p_score
        result_click = torch.sum(p_score * demand_score_candidate, dim=1)
        return result_click
    
    def deconfounder_esti(self, sess_nodes, session_last_item, item_last, item_category, adj_matrixes, sess_categories):
        """
        Args:
            sess_nodes: torch.Tensor, dtype=torch.int64,  batch_size * max_nodes_len
            session_last_item: torch.Tensor, dtype=torch.int64, batch_size
            item_last: torch.Tensor, dtype=torch.int64, batch_size
            item_category: a dic, dtype={int: list[int]}
            adj_matrixes: torch.Tensor, dtype=torch.float64, batch_size *  max_nodes_len * max_nodes_len
            sess_categories: torch.Tensor, dtype = torch.int64, batch_size * max_session_len
        Returns:
            session_last_item: torch.Tensor, dtype=torch.int64, batch_size
        """
        # KL
        user_kl_score, avarage_kl_score = get_kl_score(sess_nodes, item_last, item_category)
        node_modularity = get_modularity(adj_matrixes, sess_nodes)

        now_modularity = [0 for i in range(len(item_last))]
        P_drs = [0 for i in range(len(item_last))]
        for j in range(len(item_last)):
            temp_session_nodes = sess_nodes[j]
            temp_sess_categories = np.nan_to_num(sess_categories[j])
            if session_last_item[j]+1<len(sess_nodes[0]):
                temp_session_nodes[session_last_item[j]+1] = item_last[j]
                temp_sess_categories[session_last_item[j]+1] = item_category[item_last[j]][0]
            solid_adj_matrix, _ = self.get_adj_matrix(temp_session_nodes, sliding_size=2)

            now_modularity[j] = modularity(solid_adj_matrix, temp_sess_categories)
        mean_now_modularity = np.mean(now_modularity)
        for j in range(len(item_last)):
            if user_kl_score[j] < avarage_kl_score:
                if abs(node_modularity[j] - mean_now_modularity) < self.alpha:
                    P_dr = now_modularity[j]
                    P_dr = np.nan_to_num(P_dr)
                    P_dr = np.abs(P_dr)
                    P_drs[j] = P_dr
                else:
                    P_drs[j] = 1
            else:
                P_drs[j] = 1
        session_last_item = torch.from_numpy(np.array(session_last_item) * np.array(P_drs)).long()
        return session_last_item.to(self.device)
    
    def get_adj_matrix(self, sess_items, sliding_size=2):
        '''
        Args:
            sess_items: an item sequence
        Returns:
            adj_matrix: (solid_adj_matrix, dashed_adj_matrix), dtype = np.array
        '''
        unique_items = np.unique(sess_items)
        m_len = len(unique_items)
        solid_adj_matrix = np.zeros((m_len, m_len))
        dashed_adj_matrix = np.zeros((m_len, m_len))
        for i in np.arange(len(sess_items) - 1):  
            u = np.where(unique_items == sess_items[i])[0][0]
            v = np.where(unique_items == sess_items[i + 1])[0][0]
            solid_adj_matrix[u][v] += 1
            if sliding_size < 2: 
                raise Exception('sliding_size < 2, there are not dashed edges ')
            else:
                if i <= len(sess_items) - 1 - sliding_size:
                    for k in np.arange(2, sliding_size + 1):
                        v_1 = np.where(unique_items == sess_items[i + k])[0][0]
                        dashed_adj_matrix[u][v_1] += 1
                else:
                    pass
        return solid_adj_matrix, dashed_adj_matrix
    
    def forward(self, nodes, categories, adj, nodes_categories, session_last_item_index, candidate_category, mask_node, item_category):
        '''
        Args:
            nodes: torch.Tensor, dtype=torch.int64,  batch_size * max_nodes_len
            categories: torch.Tensor, dtype = torch.int64, batch_size * max_session_len
            adj: torch.Tensor, dtype=torch.float64, batch_size *  max_nodes_len* max_nodes_len
            nodes_categories: torch.Tensor, dtype=torch.float64, batch_size * max_nodes_len* max_session_len
            session_last_item_index: torch.Tensor, dtype=torch.int64, batch_size
            candidate_category: torch.Tensor, dtype=torch.int64, candidate item + 1
            mask_node: torch.Tensor, dtype=torch.int64, batch_size * max_nodes_len
            item_category: a dic, dtype={int: list[int]}
        Returns:
            result_click: torch.Tensor, batch_size * n_items
            catgy_click: torch.Tensor, batch_size * n_categoryies
            l2_loss: torch.Tensor, L2 regularization loss
        '''
        device = nodes.device
        session_emb = self.embedding_i(nodes)
        session_emb = self.dropout(session_emb)
        batch_size, max_nodes_len, embedding_dim_node = session_emb.shape
        if not self.ad:
            session_emb_pos = self.embedding_position(nodes)
            session_emb = session_emb + session_emb_pos
            session_emb = torch.tanh(session_emb)

        temp_node_catgy = self.normalize_matrix(nodes_categories.transpose(1, 2))
        session_emb_demand = torch.matmul(temp_node_catgy, session_emb)
        catgy_click, demand_score, demand_score_candidate, embedding_c, embedding_c_candidadte, interst_rating = self.i_demand_extractor(
                categories, candidate_category, session_emb_demand)

        self.demand_score = demand_score

        temp_node_catgy = self.normalize_matrix(nodes_categories)
        embedding_c = torch.matmul(temp_node_catgy, embedding_c)
        session_emb = torch.cat((session_emb, embedding_c), dim=-1)
        expand_session_emb = session_emb.unsqueeze(1)
        demand_session_emb = expand_session_emb.expand(batch_size, self.n_demand, max_nodes_len,
                                                       2 * embedding_dim_node)
        gnn_input = demand_session_emb 
        for l in range(self.n_gnn_layer):
            gnn_input = self.gnn_layer(gnn_input, adj, demand_score, interst_rating, nodes_categories)
        l2_loss = self.regularize(gnn_input, mask_node)
        result_click = self.recommend_demand(gnn_input, demand_score_candidate,
                                             embedding_c_candidadte, nodes, session_last_item_index,
                                             mask_node, item_category, adj, categories, batch_size)
        return result_click, catgy_click, l2_loss
    
    def normalize_matrix(self, matrix):
        """
        Normalize matrix by row
        Args:
            matrix: batch_size * max_nodes_len * max_session_len
        Returns:
            matrix: batch_size * max_nodes_len * max_session_len, normalized matrix
        """
        rowsum = matrix.sum(2).clamp(1, matrix.shape[-1])
        matrix = matrix / rowsum.unsqueeze(-1).float()
        return matrix

    def regularize(self, gnn_result, mask_nodes):
        """
        Args:
            gnn_result: batch_size * n_demand * max_nodes_len * embedding_dim_node
            mask_nodes: torch.Tensor, dtype=torch.int64, batch_size * max_nodes_len
        Returns:
            l2_loss
        """
        batch_size, _, max_nodes_len, _ = gnn_result.shape
        mask_nodes = mask_nodes.float()

        temp = torch.matmul(mask_nodes.view(batch_size, 1, 1, max_nodes_len), (gnn_result ** 2)).squeeze(-2)
        l2_loss = self.embed_l2 * temp.sum()
        return l2_loss
