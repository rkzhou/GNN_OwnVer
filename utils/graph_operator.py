import utils.datareader
import networkx as nx
import copy
import xgboost
import numpy as np
import torch
import copy
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import model.gnn_models



class Graph_self():
    def __init__(self, graph_data:utils.datareader.GraphData):
        edge_list = graph_data.adjacency.tolist()
        edge_list = self.reconstruct_edge_list(edge_list)
        self.data = copy.deepcopy(graph_data)
        self.graph = nx.Graph(edge_list)
        self.features = copy.deepcopy(graph_data.features).tolist()
        self.labels = copy.deepcopy(graph_data.labels).tolist()
        self.class_num = copy.deepcopy(graph_data.class_num)
        
    def reconstruct_edge_list(self, edge_list):
        new_edge_list = list()
        for i in range(len(edge_list[0])):
            source_node = edge_list[0][i]
            target_node = edge_list[1][i]
            node_pair = tuple([source_node, target_node])
            new_edge_list.append(node_pair)
        
        return new_edge_list
    
    def find_topk_nodes_with_adj(self, k):
        # find the top k nodes with the most edges in each class
        nodes_in_each_class = [list() for _ in range(self.class_num)]
        for i in self.data.train_nodes_index:
            nodes_in_each_class[self.labels[i]].append(i)
        
        nodes_edges = [dict() for _ in range(self.class_num)]
        topk_nodes = list()
        for class_index in range(self.class_num):
            for node_index in nodes_in_each_class[class_index]:
                node_edge_num = len(self.graph.edges(node_index))
                nodes_edges[class_index].update({node_index: node_edge_num})
        
        # sort dictionaries with the values
        new_node_edges = list()
        for class_node_edges in nodes_edges:
            class_node_edges = dict(sorted(class_node_edges.items(), key=lambda x:x[1], reverse=True))
            new_node_edges.append(class_node_edges)
        
        for i in range(self.class_num):
            topk_nodes.append(list(new_node_edges[i].keys())[:k])
        
        return topk_nodes
    

def sort_features(args, feat_num, graph_data, original_model):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Get the predictions of nodes from the original model
    predict_fn = lambda output: output.max(1, keepdim=True)[1]
    loss_fn = F.cross_entropy

    original_model.eval()
    input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    labels = graph_data.labels.to(device)
    _, output = original_model(input_data)
    original_predictions = predict_fn(output)

    chosen_feat = list()
    candidate_feat = copy.deepcopy(graph_data.features)

    for iter in range(feat_num):
        if args.benign_model == 'gcn':
            selection_model = model.gnn_models.GCN(iter+1, graph_data.class_num, hidden_dim=args.benign_hidden_dim)
        elif args.benign_model == 'sage':
            selection_model = model.gnn_models.GraphSage(iter+1, graph_data.class_num, hidden_dim=args.benign_hidden_dim)
        elif args.benign_model == 'gat':
            selection_model = model.gnn_models.GAT(iter+1, graph_data.class_num, hidden_dim=args.benign_hidden_dim)
        selection_model.to(device)
        optimizer = torch.optim.Adam(selection_model.parameters(), lr=args.benign_lr, weight_decay=args.benign_weight_decay, betas=(0.5, 0.999))
        scheduler = lr_scheduler.MultiStepLR(optimizer, args.benign_lr_decay_steps, gamma=0.1)

        feat_fidelity = dict()
        for feat_index in range(graph_data.feat_dim):
            if feat_index in chosen_feat:
                continue
            
            for module in selection_model.children():
                if isinstance(module, torch.nn.ModuleList):
                    for layer in module:
                        layer.reset_parameters()
                else:
                    module.reset_parameters()

            this_loop_feat = copy.deepcopy(chosen_feat)
            this_loop_feat.append(feat_index)
            selected_feat = candidate_feat[:, this_loop_feat]
            this_loop_input_data = selected_feat.to(device), graph_data.adjacency.to(device)
            this_loop_labels = graph_data.labels.to(device)
            selection_model.train()
            for epoch in range(args.benign_train_epochs):
                optimizer.zero_grad()
                _, output = selection_model(this_loop_input_data)
                loss = loss_fn(output[graph_data.benign_train_mask], this_loop_labels[graph_data.benign_train_mask])
                loss.backward()
                optimizer.step()
                scheduler.step()

            selection_model.eval()
            _, output = selection_model(this_loop_input_data)
            pred = predict_fn(output)
            final_pred = pred[graph_data.benign_train_mask]
            original_pred = original_predictions[graph_data.benign_train_mask]
            correct_num = 0
            for i in range(final_pred.shape[0]):
                if final_pred[i, 0] == original_pred[i, 0]:
                    correct_num += 1
                test_acc = correct_num / final_pred.shape[0] * 100
            feat_fidelity.update({feat_index: test_acc})

        feat_fidelity = sorted(feat_fidelity.items(), key=lambda x:x[1], reverse=True)
        most_important_feat = feat_fidelity[0][0]
        chosen_feat.append(most_important_feat)

    print(chosen_feat)
    return chosen_feat