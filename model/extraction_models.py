import torch
import torch_geometric.nn as nn
import torch.nn.functional as F


class SageExtract(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32], dropout=0.0):
        super(SageExtract, self).__init__()
        self.layers = torch.nn.ModuleList()

        self.layers.append(nn.SAGEConv(in_dim, hidden_dim[0]))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(nn.SAGEConv(hidden_dim[i], hidden_dim[i+1]))
        
        self.fc = nn.Linear(hidden_dim[-1], out_dim)
        #self.project_layer = nn.Linear(hidden_dim[-1], out_dim)

    def forward(self, data):
        x, edge_index = data
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        #x = self.project_layer(x)
        embedding = x
        x = self.fc(x)

        return embedding, x
    

class GcnExtract(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32], dropout=0.0):
        super(GcnExtract, self).__init__()
        self.layers = torch.nn.ModuleList()

        self.layers.append(nn.GCNConv(in_dim, hidden_dim[0]))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(nn.GCNConv(hidden_dim[i], hidden_dim[i+1]))
        
        self.fc = nn.Linear(hidden_dim[-1], out_dim)
        #self.project_layer = nn.Linear(hidden_dim[-1], out_dim)

    def forward(self, data):
        x, edge_index = data
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        #x = self.project_layer(x)
        embedding = x
        x = self.fc(x)

        return embedding, x


class GatExtract(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32], dropout=0.0):
        super(GatExtract, self).__init__()
        self.layers = torch.nn.ModuleList()

        self.layers.append(nn.GATConv(in_dim, hidden_dim[0]))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(nn.GATConv(hidden_dim[i], hidden_dim[i+1]))
        
        self.fc = nn.Linear(hidden_dim[-1], out_dim)
        #self.project_layer = nn.Linear(hidden_dim[-1], out_dim)

    def forward(self, data):
        x, edge_index = data
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        #x = self.project_layer(x)
        embedding = x
        x = self.fc(x)

        return embedding, x