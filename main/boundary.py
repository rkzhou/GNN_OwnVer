import utils.graph_operator
import copy
import torch
import torch.nn.functional as F
import torch_geometric.nn as nn
import model.gcn
import torch.optim.lr_scheduler as lr_scheduler


def poison_graph_data(graph_data, node_num, feature_num, model):
    topk_nodes = find_topk_nodes_with_loss(node_num, graph_data, model)
    topk_features = [495, 1254, 750, 581, 1336, 774, 485, 827, 38, 1110, 102, 1419, 52, 1062, 971, 1396, 149, 763, 1084, 1141, 169, 821, 915, 1135, 35, 32, 576, 863, 1195, 1193, 1237, 1090, 108, 1376, 1331, 166, 431, 263, 390, 885]

    fixed_feat = torch.rand(feature_num)
    fixed_feat = torch.where(fixed_feat<0.5, 0, 1)
    new_graph_data = copy.deepcopy(graph_data)
    
    if node_num == 0 or feature_num == 0:
        pass
    else:
        # set the value to be same
        for node_class in topk_nodes:
            for node_index in node_class:
                for i in range(feature_num):
                    new_graph_data.features[node_index][topk_features[i]] = fixed_feat[i]
    
    return new_graph_data, topk_nodes


def measure_posteriors(graph_data, posion_nodes, emb_model, clf_model):
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
    
    if posion_nodes == None:
        train_posteriors = outputs[graph_data.benign_train_mask]
        test_posteriors = outputs[graph_data.test_mask]
        # softmax = torch.nn.Softmax(dim=1)
        # train_posteriors = softmax(train_posteriors)
        # test_posteriors = softmax(test_posteriors)
        train_var = torch.sum(torch.var(train_posteriors, axis=1))
        train_var = torch.div(train_var, len(graph_data.benign_train_nodes_index))
        test_var = torch.sum(torch.var(test_posteriors, axis=1))
        test_var = torch.div(test_var, len(graph_data.test_nodes_index))
        print(train_var, test_var)
    else:
        watermark_nodes = list()
        for node_class in posion_nodes:
            watermark_nodes = watermark_nodes + node_class
        train_rest_nodes = list(set(graph_data.benign_train_nodes_index) - set(watermark_nodes))
        test_rest_nodes = list(set(graph_data.test_nodes_index) - set(watermark_nodes))

        train_posteriors = outputs[train_rest_nodes]
        test_posteriors = outputs[test_rest_nodes]
        train_var = torch.sum(torch.var(train_posteriors, axis=1))
        test_var = torch.sum(torch.var(test_posteriors, axis=1))
        print(train_var, test_var)


def find_topk_nodes_with_loss(node_num, graph_data, model):
    # By default, find the nodes with least loss
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    node_losses = [dict() for _ in range(graph_data.class_num)]
    model.eval()
    
    loss_fn = F.cross_entropy
    input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    labels = graph_data.labels.to(device)
    _, output = model(input_data)

    for node_index in graph_data.benign_train_nodes_index:
        loss = loss_fn(output[node_index], labels[node_index])
        node_losses[graph_data.labels[node_index].item()].update({node_index: loss.item()})
    
    new_node_losses = list()
    for class_node_losses in node_losses:
        class_node_losses = dict(sorted(class_node_losses.items(), key=lambda x:x[1], reverse=True))
        new_node_losses.append(class_node_losses)
    
    topk_nodes = list()
    for i in range(graph_data.class_num):
        topk_nodes.append(list(new_node_losses[i].keys())[:node_num])
    
    return topk_nodes


def find_topk_feats_with_zerro(args, feat_num, graph_data, original_model):
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
        feat_fidelity = dict()
        for feat_index in range(graph_data.feat_dim):
            if feat_index in chosen_feat:
                continue
            print('Finding feature index %d' % (feat_index))
            temp_model = model.gcn.GCN(iter+1, graph_data.class_num, hidden_dim=args.benign_hidden_dim, dropout=args.benign_dropout)
            temp_model.to(device)
            temp_optimizer = torch.optim.Adam(temp_model.parameters(), lr=args.benign_lr, weight_decay=args.benign_weight_decay, betas=(0.5, 0.999))
            temp_scheduler = lr_scheduler.MultiStepLR(temp_optimizer, args.benign_lr_decay_steps, gamma=0.1)
            
            this_loop_feat = copy.deepcopy(chosen_feat)
            this_loop_feat.append(feat_index)
            selected_feat = candidate_feat[:, this_loop_feat]
            this_loop_input_data = selected_feat.to(device), graph_data.adjacency.to(device)
            this_loop_labels = graph_data.labels.to(device)
            for epoch in range(args.benign_train_epochs):
                temp_model.train()
                temp_optimizer.zero_grad()
                _, output = temp_model(this_loop_input_data)
                loss = loss_fn(output[graph_data.benign_train_mask], this_loop_labels[graph_data.benign_train_mask])
                loss.backward()
                temp_optimizer.step()
                temp_scheduler.step()
            
            temp_model.eval()
            _, output = temp_model(this_loop_input_data)
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