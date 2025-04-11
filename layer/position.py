import torch.nn as nn

class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model, padding_idx=0)

    def forward(self, x):
        batch_size_Positional, max_nodes_len = x.shape
        return self.pe.weight[:max_nodes_len].unsqueeze(0).repeat(batch_size_Positional, 1, 1)