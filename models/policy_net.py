
import torch
import torch.nn as nn
import torch.nn.functional as F

class HungerModulatedPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, adjacency):
        super().__init__()
        self.adjacency = adjacency.coalesce()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.weights = nn.Parameter(torch.rand(self.adjacency._nnz()))
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.input_proj(x))
        adj_w = torch.sparse_coo_tensor(
            self.adjacency.indices(),
            self.weights,
            self.adjacency.shape
        )
        x = torch.sparse.mm(adj_w, x.unsqueeze(1)).squeeze(1)
        return self.output(F.relu(x))
