import torch
import torch_geometric.nn as nn
import torch.nn.functional as F


class GAT(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32], dropout=0.0):
        super(GAT, self).__init__()
        self.layers = torch.nn.ModuleList()

        self.layers.append(nn.GATConv(in_dim, hidden_dim[0]))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(nn.GATConv(hidden_dim[i], hidden_dim[i+1]))
        
        self.fc = nn.Linear(hidden_dim[-1], out_dim)

    def forward(self, data):
        x, edge_index = data
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        
        embedding = x
        x = self.fc(x)

        return embedding, x