import torch



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