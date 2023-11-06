import utils.graph_operator
import random
import copy
import torch
import torch.nn.functional as F
import torch_geometric.nn as nn
import model.gnn_models
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import pickle


def mask_graph_data(args, graph_data, model):
    mask_nodes = find_mask_nodes(args, graph_data, model)

    new_graph_data = copy.deepcopy(graph_data)
    if args.mask_node_num == 0 or args.mask_feat_num == 0:
        pass
    else:
        if args.mask_feat_type == 'random':
            mask_features = list(i for i in range(graph_data.feat_dim))
            # random.seed(args.feature_random_seed)
            random.shuffle(mask_features)
            mask_features = mask_features[:args.mask_feat_num]
        elif args.mask_feat_type == 'overall_importance':
            mask_features = find_mask_features_overall(graph_data, args.mask_feat_num)
        elif args.mask_feat_type == 'individual_importance':
            path = Path('../temp_results/feature_importances/PubMed/induc.pkl')
            if path.is_file():
                with open(path, 'rb') as f:
                    feat_importances = pickle.load(f)
            else:
                feat_importances = find_mask_features_individual(args, graph_data, model)
                with open(path, 'wb') as f:
                    pickle.dump(feat_importances, f)
            mask_features = dict()
            for node_class in mask_nodes:
                for node_index in node_class:
                    mask_features.update({node_index:feat_importances[node_index][:args.mask_feat_num]})
        else:
            raise ValueError('Invalid mask method')
        
        # flip the selected features of chosen nodes
        if args.mask_feat_type == 'random' or args.mask_feat_type == 'overall_importance':
            for node_class in mask_nodes:
                for node_index in node_class:
                    for i in range(args.mask_feat_num):
                        new_graph_data.features[node_index][mask_features[i]] = (new_graph_data.features[node_index][mask_features[i]] + 1) % 2
        elif args.mask_feat_type == 'individual_importance':
            for node_index, feat_list in mask_features.items():
                for feat_index in feat_list:
                    new_graph_data.features[node_index, feat_index] = (new_graph_data.features[node_index, feat_index] + 1) % 2
    
    return new_graph_data, mask_nodes


def measure_posteriors(args, graph_data, measure_node_class, measure_model):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    measure_model.to(device)
    measure_model.eval()
    
    if args.task_type == 'transductive':
        input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    elif args.task_type == 'inductive':
        input_data = graph_data[0].features.to(device), graph_data[0].adjacency.to(device)
    _, outputs = measure_model(input_data)
    
    measure_nodes = list()
    for node_class in measure_node_class:
        measure_nodes = measure_nodes + node_class

    node_posteriors = outputs[measure_nodes]
    softmax = torch.nn.Softmax(dim=1)
    node_posteriors = softmax(node_posteriors).detach()

    posterior_var = torch.var(node_posteriors, dim=1)
    var_mean = torch.mean(posterior_var)
    print(var_mean)


def find_mask_nodes(args, graph_data, model):
    # find the nodes from each class with least loss
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model.eval()
    
    input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    _, output = model(input_data)
    softmax = torch.nn.Softmax(dim=1)
    possibility = softmax(output)

    if args.mask_node_type == 'each_class':
        node_possibilities = [dict() for _ in range(graph_data.class_num)]
        if args.task_type == 'transductive':
            for node_index in graph_data.benign_train_nodes_index:
                node_poss = possibility[node_index].detach()
                sorted_node_poss, indices = torch.sort(node_poss, descending=True) # elements are sorted in descending order by value
                node_class_distance = sorted_node_poss[0] - sorted_node_poss[1]
                node_possibilities[graph_data.labels[node_index].item()].update({node_index: node_class_distance.item()})
    
        elif args.task_type == 'inductive':
            for node_index in range(graph_data.node_num):
                node_poss = possibility[node_index].detach()
                sorted_node_poss, indices = torch.sort(node_poss, descending=True)
                node_class_distance = sorted_node_poss[0] - sorted_node_poss[1]
                node_possibilities[graph_data.labels[node_index].item()].update({node_index: node_class_distance.item()})

        new_node_possibilities = list()
        for class_node_possibility in node_possibilities:
            class_node_possibility = dict(sorted(class_node_possibility.items(), key=lambda x:x[1], reverse=True))
            new_node_possibilities.append(class_node_possibility)
    
        topk_nodes = list()
        for i in range(graph_data.class_num):
            topk_nodes.append(list(new_node_possibilities[i].keys())[:args.mask_node_num])
    elif args.mask_node_type == 'overall':
        node_possibilities = dict()
        if args.task_type == 'transductive':
            for node_index in graph_data.benign_train_nodes_index:
                node_poss = possibility[node_index].detach()
                sorted_node_poss, indices = torch.sort(node_poss, descending=True)
                node_class_distance = sorted_node_poss[0] - sorted_node_poss[1]
                node_possibilities.update({node_index: node_class_distance.item()})
        elif args.task_type == 'inductive':
            for node_index in range(graph_data.node_num):
                node_poss = possibility[node_index].detach()
                sorted_node_poss, indices = torch.sort(node_poss, descending=True)
                node_class_distance = sorted_node_poss[0] - sorted_node_poss[1]
                node_possibilities.update({node_index: node_class_distance.item()})
        
        node_possibilities = dict(sorted(node_possibilities.items(), key=lambda x:x[1], reverse=False))
        topk_nodes = list()
        topk_nodes.append(list(node_possibilities.keys())[:args.mask_node_num])
    
    return topk_nodes


def find_mask_features_overall(graph_data, feat_num):
    X = graph_data.features.numpy()
    Y = graph_data.labels.numpy()

    dt_model = RandomForestClassifier()
    dt_model.fit(X, Y)
    feat_importance = dt_model.feature_importances_

    importance_dict = dict()
    for index, value in enumerate(feat_importance):
        importance_dict.update({index: value})
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x:x[1], reverse=True))
    topk_features = list(importance_dict.keys())[:feat_num]

    return topk_features


def find_mask_features_individual(args, graph_data, gnn_model):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    softmax = torch.nn.Softmax(dim=1)
    gnn_model.eval()
    gnn_model.to(device)

    input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    _, output = gnn_model(input_data)
    possibility = softmax(output).detach()
    var = torch.var(possibility, axis=1)
    
    search_node_list = list()
    if args.task_type == 'transductive':
        search_node_list = copy.deepcopy(graph_data.benign_train_nodes_index)
    elif args.task_type == 'inductive':
        for i in range(graph_data.node_num):
            search_node_list.append(i)

    original_variances = dict()
    for node_index in search_node_list:
        original_variances.update({node_index:var[node_index]})
    
    node_feat_importance = dict()
    for node_index in search_node_list:
        feat_var_diff = dict()
        for feat_index in range(graph_data.feat_dim):
            temp_features = copy.deepcopy(graph_data.features)
            temp_features[node_index, feat_index] = (temp_features[node_index, feat_index] + 1) % 2
            input_data = temp_features.to(device), graph_data.adjacency.to(device)
            _, output = gnn_model(input_data)
            possibility = softmax(output).detach()
            temp_var = torch.var(possibility[node_index])
            var_diff = original_variances[node_index] - temp_var
            feat_var_diff.update({feat_index:var_diff})
        feat_var_diff = dict(sorted(feat_var_diff.items(), key=lambda x:x[1], reverse=True))
        node_feat_importance.update({node_index:list(feat_var_diff.keys())})
    
    return node_feat_importance