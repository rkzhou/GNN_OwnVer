import torch
import torch_geometric.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d

class GCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32]):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()

        self.layers.append(nn.GCNConv(in_dim, hidden_dim[0]))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(nn.GCNConv(hidden_dim[i], hidden_dim[i+1]))
        
        self.fc = nn.Linear(hidden_dim[-1], out_dim)

    def forward(self, data):
        x, edge_index = data
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        
        embedding = x
        x = self.fc(x)

        return embedding, x


class GraphSage(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32]):
        super(GraphSage, self).__init__()
        self.layers = torch.nn.ModuleList()

        self.layers.append(nn.SAGEConv(in_dim, hidden_dim[0]))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(nn.SAGEConv(hidden_dim[i], hidden_dim[i+1]))
        
        self.fc = nn.Linear(hidden_dim[-1], out_dim)

    def forward(self, data):
        x, edge_index = data
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        
        embedding = x
        x = self.fc(x)

        return embedding, x


class GAT(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32]):
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


class GIN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32]):
        super(GIN, self).__init__()
        self.layers = torch.nn.ModuleList()

        self.layers.append(nn.GINConv(
            Sequential(Linear(in_dim, in_dim), BatchNorm1d(in_dim), ReLU(),
                       Linear(in_dim, hidden_dim[0]), ReLU())))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(nn.GINConv(
                Sequential(Linear(hidden_dim[i], hidden_dim[i]), BatchNorm1d(hidden_dim[i]), ReLU(),
                        Linear(hidden_dim[i], hidden_dim[i+1]), ReLU())))
        
        self.fc = nn.Linear(hidden_dim[-1], out_dim)

    def forward(self, data):
        x, edge_index = data
        for layer in self.layers:
            x = layer(x, edge_index)
        
        embedding = x
        x = self.fc(x)

        return embedding, x


class SGC(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32]):
        super(SGC, self).__init__()
        self.layers = torch.nn.ModuleList()

        self.layers.append(nn.SGConv(in_dim, hidden_dim[0], K=2))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(nn.SGConv(hidden_dim[i], hidden_dim[i+1], K=2))
        
        self.fc = nn.Linear(hidden_dim[-1], out_dim)

    def forward(self, data):
        x, edge_index = data
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        
        embedding = x
        x = self.fc(x)

        return embedding, x