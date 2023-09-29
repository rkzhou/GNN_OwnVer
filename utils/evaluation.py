import torch
import math


def get_boundary_nodes(graph_data, model, threshold, measure_part):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    predict_fn = lambda output: output.max(1, keepdim=True)[1]
    softmax = torch.nn.Softmax(dim=1)

    model.to(device)
    model.eval()
    input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    _, output = model(input_data)
    prediction = predict_fn(output).detach().cpu()
    possibility = softmax(output).detach().cpu()

    class_node_index = [list() for _ in range(graph_data.class_num)]
    if measure_part == 'train':
        for i in range(len(graph_data.benign_train_nodes_index)):
            node_index = graph_data.benign_train_nodes_index[i]
            class_node_index[prediction[node_index]].append(node_index)
    elif measure_part == 'test':
        for i in range(len(graph_data.test_nodes_index)):
            node_index = graph_data.test_nodes_index[i]
            class_node_index[prediction[node_index]].append(node_index)
    
    # calculate the center of each training class
    class_centers = torch.zeros([graph_data.class_num, graph_data.class_num])
    for i in range(graph_data.class_num):
        node_class = class_node_index[i]
        possibility_center = torch.zeros([1, graph_data.class_num])
        for node_index in node_class:
            possibility_center += possibility[node_index]
        possibility_center /= len(node_class)
        class_centers[i] = possibility_center
    
    # get the node index within the boundary margin between each two classes
    boundary_node_index = set()
    for i in range(graph_data.class_num):
        for j in range(graph_data.class_num):
            if i >= j:
                continue
            A_center = class_centers[i]
            B_center = class_centers[j]
            AB_bound = (A_center + B_center) / 2.0
            A_upper_bound = AB_bound[i] + threshold
            B_upper_bound = AB_bound[j] + threshold
            
            A_class_nodes = class_node_index[i]
            B_class_nodes = class_node_index[j]

            # iterate over the node index in A,B class and check which nodes whose possibility values are smaller than upper bound
            for node_index in A_class_nodes:
                if possibility[node_index][i] <= A_upper_bound:
                    boundary_node_index.add(node_index)
            
            for node_index in B_class_nodes:
                if possibility[node_index][j] <= B_upper_bound:
                    boundary_node_index.add(node_index)
    
    boundary_node_index = list(boundary_node_index)
    return boundary_node_index


def eval_node_label_change(graph_data, original_model, mask_model, portion_num):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    original_model.eval()
    mask_model.eval()
    original_model.to(device)
    mask_model.to(device)

    predict_fn = lambda output: output.max(1, keepdim=True)[1]
    training_node_num = len(graph_data.benign_train_nodes_index)
    testing_node_num = len(graph_data.test_nodes_index)

    train_portion_node_num = math.ceil(training_node_num / portion_num)
    test_portion_node_num = math.ceil(testing_node_num / portion_num)

    input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    _, original_output = original_model(input_data)
    _, mask_output = mask_model(input_data)
    original_pred = predict_fn(original_output).detach().cpu()
    mask_pred = predict_fn(mask_output).detach().cpu()

    # find node index in each portion by possibility in training dataset
    softmax = torch.nn.Softmax(dim=1)
    possibility = softmax(original_output).detach().cpu()
    train_node_possibilities, test_node_possibilities = dict(), dict()
    for node_index in graph_data.benign_train_nodes_index:
        node_poss = possibility[node_index]
        sorted_node_poss, indices = torch.sort(node_poss, descending=True)
        node_class_distance = sorted_node_poss[0] - sorted_node_poss[1]
        train_node_possibilities.update({node_index: node_class_distance.item()})
        
    train_node_possibilities = dict(sorted(train_node_possibilities.items(), key=lambda x:x[1], reverse=True))
    train_each_portion_node_index = dict()
    for i in range(portion_num):
        start_pos = i * train_portion_node_num
        end_pos = (i + 1) * train_portion_node_num
        train_each_portion_node_index.update({i:list(train_node_possibilities.keys())[start_pos:end_pos]})
    
    # find node index in each portion by possibility in testing dataset
    for node_index in graph_data.test_nodes_index:
        node_poss = possibility[node_index]
        sorted_node_poss, indices = torch.sort(node_poss, descending=True)
        node_class_distance = sorted_node_poss[0] - sorted_node_poss[1]
        test_node_possibilities.update({node_index: node_class_distance.item()})
        
    test_node_possibilities = dict(sorted(test_node_possibilities.items(), key=lambda x:x[1], reverse=True))
    test_each_portion_node_index = dict()
    for i in range(portion_num):
        start_pos = i * test_portion_node_num
        end_pos = (i + 1) * test_portion_node_num
        test_each_portion_node_index.update({i:list(test_node_possibilities.keys())[start_pos:end_pos]})
    
    # check how many nodes' lables are changed in each portion
    each_portion_train_nodes_changed_num, each_portion_test_nodes_changed_num = list(), list()
    original_each_portion_train_nodes_labels_distribute, original_each_portion_test_nodes_labels_distribute = dict(), dict() #True num, False num
    mask_each_portion_train_nodes_labels_distribute, mask_each_portion_test_nodes_labels_distribute = dict(), dict()
    for portion_index, node_list in train_each_portion_node_index.items():
        changed_num = 0
        original_true_num, original_false_num = 0, 0
        mask_true_num, mask_false_num = 0, 0
        for node_index in node_list:
            if original_pred[node_index] != mask_pred[node_index]:
                changed_num += 1

            if original_pred[node_index] == graph_data.labels[node_index]:
                original_true_num += 1
            else:
                original_false_num += 1
            
            if mask_pred[node_index] == graph_data.labels[node_index]:
                mask_true_num += 1
            else:
                mask_false_num += 1
        
        each_portion_train_nodes_changed_num.append(changed_num)
        original_each_portion_train_nodes_labels_distribute.update({portion_index:[original_true_num, original_false_num]})
        mask_each_portion_train_nodes_labels_distribute.update({portion_index:[mask_true_num, mask_false_num]})
    
    for portion_index, node_list in test_each_portion_node_index.items():
        changed_num = 0
        original_true_num, original_false_num = 0, 0
        mask_true_num, mask_false_num = 0, 0
        for node_index in node_list:
            if original_pred[node_index] != mask_pred[node_index]:
                changed_num += 1

            if original_pred[node_index] == graph_data.labels[node_index]:
                original_true_num += 1
            else:
                original_false_num += 1
            
            if mask_pred[node_index] == graph_data.labels[node_index]:
                mask_true_num += 1
            else:
                mask_false_num += 1
        
        each_portion_test_nodes_changed_num.append(changed_num)
        original_each_portion_test_nodes_labels_distribute.update({portion_index:[original_true_num, original_false_num]})
        mask_each_portion_test_nodes_labels_distribute.update({portion_index:[mask_true_num, mask_false_num]})
    
    print(each_portion_train_nodes_changed_num)
    print(original_each_portion_train_nodes_labels_distribute)
    print(mask_each_portion_train_nodes_labels_distribute)