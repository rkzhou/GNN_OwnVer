import os
import torch
import torch_geometric.datasets as dt
import random
import math
import copy

def get_data(args):
    if args.dataset == 'Cora_ML' or args.dataset == 'Citeseer' or args.dataset == 'DBLP' or args.dataset == 'PubMed':
        dataset = dt.CitationFull(args.data_path, args.dataset)
        data_path = args.data_path + '/' + (args.dataset).lower() + '/processed/data.pt'
    elif args.dataset == 'Coauthor':
        dataset = dt.Coauthor(args.data_path, 'Physics')
        data_path = args.data_path + '/' + 'Physics' + '/processed/data.pt'
    elif args.dataset == 'Amazon':
        dataset = dt.Amazon(args.data_path, 'Photo')
        data_path = args.data_path + '/' + 'Photo' + '/processed/data.pt'
    
    data = torch.load(data_path)
    
    return data


class GraphData(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.features = data[0]['x']
        self.adjacency = data[0]['edge_index']
        self.labels = data[0]['y']
        self.node_num = len(self.labels)
        self.feat_dim = len(self.features[0])
        self.benign_train_mask = None
        self.extraction_train_mask = None
        self.test_mask = None
        self.set_adj_mat()
        self.get_class_num()
        self.set_mask(args)
    
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

    def set_mask(self, args):
        all_nodes_index = list(i for i in range(self.node_num))
        self.benign_train_nodes_index, self.extraction_train_nodes_index, self.test_nodes_index = list(), list(), list()
        
        each_class_nodes_index = [list() for _ in range(self.class_num)]
        for i in range(self.node_num):
            each_class_nodes_index[self.labels[i]].append(i)
        for i in range(self.class_num):
            # random.seed(args.random_seed)
            random.shuffle(each_class_nodes_index[i])
            class_node_num = len(each_class_nodes_index[i])
            benign_train_size = math.floor(class_node_num * args.benign_train_ratio)
            extraction_train_size = math.floor(class_node_num * (1.0-args.benign_train_ratio) * args.extraction_ratio)
            test_size = class_node_num - benign_train_size - extraction_train_size
            # print(benign_train_size, extraction_train_size, test_size)

            self.benign_train_nodes_index += each_class_nodes_index[i][:benign_train_size]
            self.extraction_train_nodes_index += each_class_nodes_index[i][benign_train_size:(benign_train_size+extraction_train_size)]
            self.test_nodes_index += each_class_nodes_index[i][(benign_train_size+extraction_train_size):]
        

        self.benign_train_mask, self.extraction_train_mask, self.test_mask = torch.zeros(self.node_num), torch.zeros(self.node_num), torch.zeros(self.node_num)
        self.benign_train_mask[self.benign_train_nodes_index] = 1
        self.extraction_train_mask[self.extraction_train_nodes_index] = 1
        self.test_mask[self.test_nodes_index] = 1
        self.benign_train_mask = self.benign_train_mask.bool()
        self.extraction_train_mask = self.extraction_train_mask.bool()
        self.test_mask = self.test_mask.bool()

    def __len__(self):
        return self.node_num
    
    def __getitem__(self, index):
        return [self.features[index], self.adj_matrix[index], self.labels[index]]


class DistanceData(torch.utils.data.Dataset):
    def __init__(self, label0_data_list, label1_data_list):
        self.label0_data_list = copy.deepcopy(label0_data_list)
        self.label1_data_list = copy.deepcopy(label1_data_list)
        self.concat_data()
        label0_data_label = [0 for _ in range(self.label0_data.shape[0])]
        label1_data_label = [1 for _ in range(self.label1_data.shape[0])]
        self.label = label0_data_label + label1_data_label
        self.label = torch.as_tensor(self.label)
    
    def concat_data(self):
        self.label0_data = None
        self.label1_data = None
        for data_index in range(len(self.label0_data_list)):
            if data_index == 0:
                self.label0_data = self.label0_data_list[data_index]
                self.label1_data = self.label1_data_list[data_index]
            else:
                self.label0_data = torch.cat((self.label0_data, self.label0_data_list[data_index]), 0)
                self.label1_data = torch.cat((self.label1_data, self.label1_data_list[data_index]), 0)
        
        self.data = torch.cat((self.label0_data, self.label1_data), 0)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data[index, :], self.label[index]

if __name__ == '__main__':
    pass