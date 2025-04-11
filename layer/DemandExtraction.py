import math
import torch as t
import torch.nn as nn
from torch.nn import Module

class DExtraction(Module):
    def __init__(self, n_demand, n_categories, embedding_dim_c, hidden_size, drop=0.25):
        """
        Args:
            n_demand: int
            n_categories: int
            embedding_dim_c: int
            hidden_size: int
        """
        super(DExtraction, self).__init__()
        self.n_categories = n_categories
        self.n_demand = n_demand
        self.hidden_size = hidden_size

        self.embedding_c = nn.Embedding(n_categories, embedding_dim_c, padding_idx=0)
        self.dropout = nn.Dropout(drop)
        self.demand_linear = nn.Linear(embedding_dim_c, n_demand * hidden_size, bias=False)
        self.key_linear = nn.Linear(embedding_dim_c, hidden_size)
        self.catgy_linear = nn.Linear(embedding_dim_c, n_demand * hidden_size)

    def weight_init(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        self.w_score.data.uniform_(std)
        return None
                      
    def compute_demand_score(self, hidden_demand_agg, hidden_key, hidden_key_candidate):
        """
        Args:
            hidden_demand_agg: torch.Tensor, batch_size * n_demand * hidden_size
            hidden_key: torch.Tensor, batch_size * max_session_len * hidden_size
            hidden_key_candidate: torch.Tensor, n_items * hidden_size
        Returns:
            demand_score: torch.Tensor, batch_size  * n_demand * max_session_len
            demand_score_candidate: torch.Tensor,  batch_size * n_demand * n_items
        """
        batch_size, max_session_len = hidden_key.shape[0], hidden_key.shape[1]

        hidden_key = hidden_key.view(batch_size, max_session_len, 1, self.hidden_size, 1)
        hidden_demand_agg = hidden_demand_agg.view(batch_size, 1, self.n_demand, 1, self.hidden_size)
        demand_score = t.matmul(hidden_demand_agg, hidden_key) / (self.hidden_size ** (1 / 2))
        demand_score = demand_score.view(batch_size, self.n_demand, max_session_len)

        hidden_key_candidate = hidden_key_candidate.view(1, 1, -1, self.hidden_size)
        hidden_demand_agg = hidden_demand_agg.transpose(-1, -2).squeeze(1)
        demand_score_candidate = t.matmul(hidden_key_candidate, hidden_demand_agg) / (self.hidden_size ** (1 / 2))
        demand_score_candidate = demand_score_candidate.squeeze(-1)

        return demand_score, demand_score_candidate
    
    def forward(self, input, candidate_pool_category, embedding_i):
        """
        Args:
            input: torch.Tensor, dtype = torch.int64, batch_size * max_session_len
            candidate_pool_category: torch.Tensor, dtype=torch.int64, candidate item + 1
            embedding_i: torch.Tensor, batch_size * max_session_len * hidden_dim
        Returns:
            catgy_click: torch.Tensor, batch_size * n_categoryies
            demand_score: demand_scores: batch_size * n_demand * max_session_len
            demand_score_candidate: torch.Tensor, dtype=torch.float32  batch_size * n_demand * n_items
            embedding: torch.Tensor, dtype=torch.float32, batch_size * max_session_len * embedding_dim_c
            embedding_candidate: torch.Tensor, dtype=torch.float32, n_items * embedding_dim_c
            interst_rating: torch.Tensor, dtype=torch.float32, batch_size * n_interest * max_session_len
        """
        batch_size, max_session_len = input.shape
        embedding_c = self.embedding_c(input)
        embedding = embedding_c + embedding_i
        embedding = self.dropout(embedding)

        embedding_candidate = self.embedding_c(candidate_pool_category)
        hidden_key = self.key_linear(embedding)
        hidden_demand = self.demand_linear(embedding).view(batch_size, max_session_len, self.n_demand, self.hidden_size)
        hidden_demand_agg = hidden_demand.exp().sum(1).log()
        hidden_key_candidate = self.key_linear(embedding_candidate)

        demand_score, demand_score_candidate = self.compute_demand_score(hidden_demand_agg, hidden_key, hidden_key_candidate)
        demand_score = t.sigmoid(demand_score)
        demand_score_candidate = t.sigmoid(demand_score_candidate)
        
        return demand_score, demand_score_candidate, embedding, embedding_candidate