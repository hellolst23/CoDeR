import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model, padding_idx=0)

    def forward(self, x):
        batch_size_Positional, max_nodes_len = x.shape
        # print(self.pe.weight[:max_nodes_len].unsqueeze(0).repeat(batch_size_Positional, 1, 1).shape)
        return self.pe.weight[:max_nodes_len].unsqueeze(0).repeat(batch_size_Positional, 1, 1)