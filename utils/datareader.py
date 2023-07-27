import os
import torch
import torch_geometric.datasets as dt
import random
import math

def get_data(args):
    if args.dataset == 'Cora':
        dataset = dt.Planetoid(args.data_path, args.dataset)
        data_path = args.data_path + '/' + args.dataset + '/processed/data.pt'
    
    data = torch.load(data_path)
    
    return data


class GraphData(torch.utils.data.Dataset):
    def __init__(self, data, split_ratio):
        self.features = data[0]['x']
        self.adjacency = data[0]['edge_index']
        self.labels = data[0]['y']
        self.node_num = len(self.labels)
        self.feat_dim = len(self.features[0])
        self.train_mask = None
        self.test_mask = None
        self.set_adj_mat()
        self.get_class_num()
        self.set_mask(split_ratio)
    
    def set_adj_mat(self):
        self.adj_matrix = torch.zeros([self.node_num, self.node_num])
        for i in range(self.node_num):
            source_node = self.adjacency[0][i]
            target_node = self.adjacency[1][i]
            self.adj_matrix[source_node, target_node] = 1
    
    def get_class_num(self):
        labels = self.labels.tolist()
        labels = set(labels)
        self.class_num = len(labels)

    def set_mask(self, split_ratio):
        all_nodes_index = list(i for i in range(self.node_num))
        random.shuffle(all_nodes_index)
        train_size = math.floor(self.node_num * split_ratio)
        test_size = self.node_num - train_size
        self.train_nodes_index = all_nodes_index[:train_size]
        self.test_nodes_index = all_nodes_index[train_size:]
        self.train_mask, self.test_mask = torch.zeros(self.node_num), torch.zeros(self.node_num)
        self.train_mask[self.train_nodes_index] = 1
        self.test_mask[self.test_nodes_index] = 1
        self.train_mask = self.train_mask.bool()
        self.test_mask = self.test_mask.bool()

    def __len__(self):
        return self.node_num
    
    def __getitem__(self, index):
        return [self.features[index], self.adj_matrix[index], self.labels[index]]



if __name__ == '__main__':
    pass