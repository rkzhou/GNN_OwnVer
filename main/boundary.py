import utils.graph_operator
import random
import copy
import torch
import torch.nn.functional as F
import torch_geometric.nn as nn
import model.gnn_models
import torch.optim.lr_scheduler as lr_scheduler


def poison_graph_data(graph_data, node_num, feature_num, model, pick_nodes_type):
    topk_nodes = find_topk_nodes_with_loss(graph_data, node_num, model, pick_nodes_type)
    topk_features = list(i for i in range(graph_data.feat_dim))
    random.shuffle(topk_features)
    topk_features = topk_features[:feature_num]
    # topk_features = [495, 1254, 750, 581, 1336, 774, 485, 827, 38, 1110, 102, 1419, 52, 1062, 971, 1396, 149, 
    #                  763, 1084, 1141, 169, 821, 915, 1135, 35, 32, 576, 863, 1195, 1193, 1237, 1090, 108, 1376, 1331, 166, 431, 263, 390, 885]
    # topk_features = topk_features[:feature_num]

    new_graph_data = copy.deepcopy(graph_data)
    if node_num == 0 or feature_num == 0:
        pass
    else:
        # mask the features of chosen nodes
        for node_class in topk_nodes:
            for node_index in node_class:
                fixed_feat = torch.rand(feature_num)
                fixed_feat = torch.where(fixed_feat<0.5, 0, 1)
                for i in range(feature_num):
                    new_graph_data.features[node_index][topk_features[i]] = fixed_feat[i]
    
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
        train_posteriors = softmax(train_posteriors)
        test_posteriors = softmax(test_posteriors)
        
        # train_entropy = torch.sum(torch.sum(torch.special.entr(train_posteriors), dim=1))
        # test_entropy = torch.sum(torch.sum(torch.special.entr(test_posteriors), dim=1))
        # print(train_entropy, test_entropy)
        
        train_var = torch.sum(torch.var(train_posteriors, axis=1))
        train_var = torch.div(train_var, len(graph_data.benign_train_nodes_index))
        test_var = torch.sum(torch.var(test_posteriors, axis=1))
        test_var = torch.div(test_var, len(graph_data.test_nodes_index))
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
            loss = loss_fn(output[node_index], labels[node_index])
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
            loss = loss_fn(output[node_index], labels[node_index])
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
    loss_fn = F.cross_entropy
    input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    labels = graph_data.labels.to(device)
    _, output = model(input_data)
    softmax = torch.nn.Softmax(dim=1)
    possibility = softmax(output)

    if type == 'each_class':
        node_possibilities = [dict() for _ in range(graph_data.class_num)]
        for node_index in graph_data.benign_train_nodes_index:
            node_poss = possibility[node_index]
            sorted_node_poss, indices = torch.sort(node_poss, descending=True)
            node_class_distance = sorted_node_poss[0] - sorted_node_poss[1]
            node_possibilities[graph_data.labels[node_index].item()].update({node_index: node_class_distance.item()})
    
        new_node_possibilities = list()
        for class_node_possibility in node_possibilities:
            class_node_possibility = dict(sorted(class_node_possibility.items(), key=lambda x:x[1], reverse=False))
            new_node_possibilities.append(class_node_possibility)
    
        topk_nodes = list()
        for i in range(graph_data.class_num):
            topk_nodes.append(list(new_node_possibilities[i].keys())[:node_num])
    elif type == 'overall':
        node_possibilities = dict()
        for node_index in graph_data.benign_train_nodes_index:
            node_poss = possibility[node_index]
            sorted_node_poss, indices = torch.sort(node_poss, descending=True)
            node_class_distance = sorted_node_poss[0] - sorted_node_poss[1]
            node_possibilities.update({node_index: node_class_distance.item()})
        
        node_possibilities = dict(sorted(node_possibilities.items(), key=lambda x:x[1], reverse=False))
        topk_nodes = list()
        topk_nodes.append(list(node_possibilities.keys())[:node_num])
    
    return topk_nodes