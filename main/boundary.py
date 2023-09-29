import utils.graph_operator
import random
import copy
import torch
import torch.nn.functional as F
import torch_geometric.nn as nn
import model.gnn_models
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.ensemble import RandomForestClassifier


def mask_graph_data(args, graph_data, model):
    topk_nodes = find_topk_nodes_with_possibility(graph_data, args.mask_node_num, model, args.mask_node_type)

    new_graph_data = copy.deepcopy(graph_data)
    if args.mask_node_num == 0 or args.mask_feat_num == 0:
        pass
    else:
        if args.mask_feat_type == 'random':
            topk_features = list(i for i in range(graph_data.feat_dim))
            random.shuffle(topk_features)
            topk_features = topk_features[:args.mask_feat_num]
        elif args.mask_feat_type == 'overall_importance':
            topk_features = find_topk_features_overall(graph_data, args.mask_feat_num)
        elif args.mask_feat_type == 'individual_importance':
            topk_features = find_topk_features_individual(graph_data, model, topk_nodes, args.mask_feat_num)
        
        # flip the selected features of chosen nodes
        if args.mask_feat_type == 'random' or args.mask_feat_type == 'overall_importance':
            for node_class in topk_nodes:
                for node_index in node_class:
                    for i in range(args.mask_feat_num):
                        new_graph_data.features[node_index][topk_features[i]] = (new_graph_data.features[node_index][topk_features[i]] + 1) % 2
        elif args.mask_feat_type == 'individual_importance':
            for node_index, feat_list in topk_features.items():
                for feat_index in feat_list:
                    new_graph_data.features[node_index, feat_index] = (new_graph_data.features[node_index, feat_index] + 1) % 2
    
    return new_graph_data, topk_nodes


def measure_posteriors(graph_data, specific_nodes, emb_model, clf_model=None):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    emb_model.to(device)
    emb_model.eval()
    #clf_model.eval()
    
    input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    embeddings, outputs = emb_model(input_data)
    #outputs = F.softmax(outputs, dim=1)
    #outputs = clf_model(embeddings.detach())
    
    if specific_nodes == None:
        train_posteriors = outputs[graph_data.benign_train_mask]
        test_posteriors = outputs[graph_data.test_mask]
        
        softmax = torch.nn.Softmax(dim=1)
        train_posteriors = softmax(train_posteriors).detach().cpu()
        test_posteriors = softmax(test_posteriors).detach().cpu()
        
        train_var = torch.var(train_posteriors, axis=1)
        train_var = torch.mean(train_var)
        test_var = torch.var(test_posteriors, axis=1)
        test_var = torch.mean(test_var)
        print(train_var, test_var)
    else:
        measure_nodes = list()
        for node_class in specific_nodes:
            measure_nodes = measure_nodes + node_class
        # train_rest_nodes = list(set(graph_data.benign_train_nodes_index) - set(watermark_nodes))
        # test_rest_nodes = list(set(graph_data.test_nodes_index) - set(watermark_nodes))

        node_posteriors = outputs[measure_nodes]
        softmax = torch.nn.Softmax(dim=1)
        node_posteriors = softmax(node_posteriors).detach().cpu()

        posterior_var = torch.var(node_posteriors, dim=1)
        var_mean = torch.mean(posterior_var)
        print(var_mean)
        
        return node_posteriors


def find_topk_nodes_with_loss(graph_data, node_num, model, type):
    # find the nodes from each class with least loss
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model.eval()
    loss_fn = F.cross_entropy
    input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    labels = graph_data.labels.to(device)
    _, output = model(input_data)

    if type == 'each_class':
        node_losses = [dict() for _ in range(graph_data.class_num)]
        for node_index in graph_data.benign_train_nodes_index:
            loss = loss_fn(output[node_index], labels[node_index]).detach().cpu()
            node_losses[graph_data.labels[node_index].item()].update({node_index: loss.item()})
    
        new_node_losses = list()
        for class_node_losses in node_losses:
            class_node_losses = dict(sorted(class_node_losses.items(), key=lambda x:x[1], reverse=False)) # False will sort ascending, True will sort descending.
            new_node_losses.append(class_node_losses)
    
        topk_nodes = list()
        for i in range(graph_data.class_num):
            topk_nodes.append(list(new_node_losses[i].keys())[:node_num])
    elif type == 'overall':
        node_losses = dict()
        for node_index in graph_data.benign_train_nodes_index:
            loss = loss_fn(output[node_index], labels[node_index]).detach().cpu()
            node_losses.update({node_index: loss.item()})
        
        node_losses = dict(sorted(node_losses.items(), key=lambda x:x[1], reverse=False))
        topk_nodes = list()
        topk_nodes.append(list(node_losses.keys())[:node_num])
    
    return topk_nodes


def find_topk_nodes_with_possibility(graph_data, node_num, model, type):
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

    if type == 'each_class':
        node_possibilities = [dict() for _ in range(graph_data.class_num)]
        for node_index in graph_data.benign_train_nodes_index:
            node_poss = possibility[node_index].detach().cpu()
            sorted_node_poss, indices = torch.sort(node_poss, descending=True) # elements are sorted in descending order by value
            node_class_distance = sorted_node_poss[0] - sorted_node_poss[1]
            node_possibilities[graph_data.labels[node_index].item()].update({node_index: node_class_distance.item()})
    
        new_node_possibilities = list()
        for class_node_possibility in node_possibilities:
            class_node_possibility = dict(sorted(class_node_possibility.items(), key=lambda x:x[1], reverse=True))
            new_node_possibilities.append(class_node_possibility)
    
        topk_nodes = list()
        for i in range(graph_data.class_num):
            topk_nodes.append(list(new_node_possibilities[i].keys())[:node_num])
    elif type == 'overall':
        node_possibilities = dict()
        for node_index in graph_data.benign_train_nodes_index:
            node_poss = possibility[node_index].detach().cpu()
            sorted_node_poss, indices = torch.sort(node_poss, descending=True)
            node_class_distance = sorted_node_poss[0] - sorted_node_poss[1]
            node_possibilities.update({node_index: node_class_distance.item()})
        
        node_possibilities = dict(sorted(node_possibilities.items(), key=lambda x:x[1], reverse=False))
        topk_nodes = list()
        topk_nodes.append(list(node_possibilities.keys())[:node_num])
    
    return topk_nodes


def find_topk_features_overall(graph_data, feat_num):
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


def find_topk_features_individual(graph_data, gnn_model, node_list, feat_num):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    softmax = torch.nn.Softmax(dim=1)
    gnn_model.eval()
    gnn_model.to(device)

    input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    _, output = gnn_model(input_data)
    possibility = softmax(output).detach().cpu()
    var = torch.var(possibility, axis=1)

    temp_node_list = list()
    for node_class in node_list:
        temp_node_list += node_class
    
    original_variances = dict()
    for node_index in temp_node_list:
        original_variances.update({node_index:var[node_index]})
    
    node_selected_feat = dict()
    for node_index in temp_node_list:
        feat_var_diff = dict()
        for feat_index in range(graph_data.feat_dim):
            temp_features = copy.deepcopy(graph_data.features)
            temp_features[node_index, feat_index] = (temp_features[node_index, feat_index] + 1) % 2
            input_data = temp_features.to(device), graph_data.adjacency.to(device)
            _, output = gnn_model(input_data)
            possibility = softmax(output).detach().cpu()
            temp_var = torch.var(possibility[node_index])
            var_diff = original_variances[node_index] - temp_var
            feat_var_diff.update({feat_index:var_diff})
        feat_var_diff = dict(sorted(feat_var_diff.items(), key=lambda x:x[1], reverse=True))
        node_selected_feat.update({node_index:list(feat_var_diff.keys())[:feat_num]})
    
    return node_selected_feat