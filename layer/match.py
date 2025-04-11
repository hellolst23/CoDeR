from layer import *
from util.utils import timefn

class Match(nn.Module): 
    def __init__(self, hidden_size, batchNorm = 'feature', n_demand=2, rs='dot'):
        super(Match, self).__init__()
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
            self.nn_linear2 = nn.Linear(hidden_size,1)
        elif self.rs == 'dot':
            self.nn_linear1 = nn.Linear(2*hidden_size, hidden_size)

    def batch_norm_custom(self, hidden):
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
        if self.rs == 'mlp':
            sess_represenation = sess_represenation.unsqueeze(2).repeat(1, 1, candidate_items_embedding.shape[2], 1)
            hidden = torch.cat((sess_represenation, candidate_items_embedding), dim=-1)
            hidden = self.nn_linear1(hidden)
            hidden = self.batch_norm_custom(hidden)
            score = self.nn_linear2(self.act_func(hidden)).squeeze(-1)

        elif self.rs == 'dot':
            hidden = sess_represenation[:,:,:self.hidden_size]
            hidden = hidden.unsqueeze(-2)
            score = torch.matmul(candidate_items_embedding, hidden.transpose(-1, -2)).squeeze(-1)
        return score