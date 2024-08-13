
# @File    : test_config.py
# @Software: PyCharm

# Model config
n_demand = 2  # Demand number
embedding_dim_i = 10  # emb dim for item
embedding_dim_c = 10  # emb dim for categories
hidden_size = 15  # hidden size for demand extraction
dashed_order = 2  # order of dashed edges
bias = False
non_linear = 'relu'
n_gnn_layer = 2  # Number of the gnn layer
graph_aggregation = 'mean'  # method to aggregate the graph, lstm, mean, sum...

batchNorm = 'feature' # demand, feature, None
non_linear_demand_score = 'relu' # relu, sigmoid