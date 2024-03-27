import torch
import copy
import utils.config
import utils.datareader
import model.gnn_models
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from utils.graph_operator import split_subgraph
from tqdm import tqdm
from utils.config import parse_args
from pathlib import Path
import random


def transductive_train(args, model_save_path, graph_data, process):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if graph_data == None:
        data = utils.datareader.get_data(args)
        gdata = utils.datareader.GraphData(data, args)
    else:
        gdata = graph_data
    
    path = Path(model_save_path)

    # training
    loss_fn = torch.nn.CrossEntropyLoss()
    predict_fn = lambda output: output.max(1, keepdim=True)[1]


    if path.is_file():
        gnn_model = torch.load(model_save_path)
    else:
        if args.benign_model == 'gcn':
            gnn_model = model.gnn_models.GCN(gdata.feat_dim, gdata.class_num, hidden_dim=args.benign_hidden_dim)
        elif args.benign_model == 'sage':
            gnn_model = model.gnn_models.GraphSage(gdata.feat_dim, gdata.class_num, hidden_dim=args.benign_hidden_dim)
        elif args.benign_model == 'gat':
            gnn_model = model.gnn_models.GAT(gdata.feat_dim, gdata.class_num, hidden_dim=args.benign_hidden_dim)
        elif args.benign_model == 'gin':
            gnn_model = model.gnn_models.GIN(gdata.feat_dim, gdata.class_num, hidden_dim=args.benign_hidden_dim)
        elif args.benign_model == 'sgc':
            gnn_model = model.gnn_models.SGC(gdata.feat_dim, gdata.class_num, hidden_dim=args.benign_hidden_dim)

        gnn_model.to(device)

        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.benign_lr)

        last_train_acc = 0.0
        if process == 'test':
            train_nodes_index = [i for i in range(gdata.node_num)]
            random.shuffle(train_nodes_index)
            train_nodes_index = train_nodes_index[:len(gdata.target_nodes_index)]
        
        for epoch in range(args.benign_train_epochs):
            gnn_model.train()
            optimizer.zero_grad()
            input_data = gdata.features.to(device), gdata.adjacency.to(device)
            labels = gdata.labels.to(device)
            _, output = gnn_model(input_data)
            if process == 'test':
                loss = loss_fn(output[train_nodes_index], labels[train_nodes_index])
            else:
                loss = loss_fn(output[gdata.target_nodes_index], labels[gdata.target_nodes_index])
            loss.backward()
            optimizer.step()

            train_correct_num = 0
            if (epoch + 1) % 50 == 0:
                _, output = gnn_model(input_data)
                pred = predict_fn(output)
                if process == 'test':
                    train_pred = pred[train_nodes_index]
                    train_labels = gdata.labels[train_nodes_index]
                else:
                    train_pred = pred[gdata.target_nodes_index]
                    train_labels = gdata.labels[gdata.target_nodes_index]
                
                for i in range(train_pred.shape[0]):
                    if train_pred[i, 0] == train_labels[i]:
                        train_correct_num += 1
                train_acc = train_correct_num / train_pred.shape[0] * 100

                if last_train_acc == 0.0:
                    last_train_acc = train_acc
                else:
                    train_acc_diff = (train_acc - last_train_acc) / last_train_acc * 100
                    if train_acc_diff <= 0.5: #0.5%
                        break
                    else:
                        last_train_acc = train_acc
        
        torch.save(gnn_model, model_save_path)
    
    test_correct_num = 0
    gnn_model.eval()
    input_data = gdata.features.to(device), gdata.adjacency.to(device)
    _, output = gnn_model(input_data)
    pred = predict_fn(output)
    test_pred = pred[gdata.test_nodes_index]
    test_labels = gdata.labels[gdata.test_nodes_index]
    for i in range(test_pred.shape[0]):
        if test_pred[i, 0] == test_labels[i]:
            test_correct_num += 1
    test_acc = test_correct_num / test_pred.shape[0] * 100
    save_test_acc = round(test_acc, 3)
    
    return gdata, gnn_model, save_test_acc


def inductive_train(args, model_save_path, graph_data, process):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if graph_data == None:
        data = utils.datareader.get_data(args)
        gdata = utils.datareader.GraphData(data, args)
        target_graph_data, shadow_graph_data, attacker_graph_data, test_graph_data = split_subgraph(gdata)
    else:
        target_graph_data, shadow_graph_data, attacker_graph_data, test_graph_data = graph_data

    loss_fn = torch.nn.CrossEntropyLoss()
    predict_fn = lambda output: output.max(1, keepdim=True)[1]
    path = Path(model_save_path)
    if path.is_file():
        gnn_model = torch.load(model_save_path)
    else:
        if args.benign_model == 'gcn':
            gnn_model = model.gnn_models.GCN(target_graph_data.feat_dim, target_graph_data.class_num, hidden_dim=args.benign_hidden_dim)
        elif args.benign_model == 'sage':
            gnn_model = model.gnn_models.GraphSage(target_graph_data.feat_dim, target_graph_data.class_num, hidden_dim=args.benign_hidden_dim)
        elif args.benign_model == 'gat':
            gnn_model = model.gnn_models.GAT(target_graph_data.feat_dim, target_graph_data.class_num, hidden_dim=args.benign_hidden_dim)
        elif args.benign_model == 'gin':
            gnn_model = model.gnn_models.GIN(target_graph_data.feat_dim, target_graph_data.class_num, hidden_dim=args.benign_hidden_dim)
        elif args.benign_model == 'sgc':
            gnn_model = model.gnn_models.SGC(target_graph_data.feat_dim, target_graph_data.class_num, hidden_dim=args.benign_hidden_dim)

        gnn_model.to(device)


        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.benign_lr)

        last_train_acc = 0.0

        if process == 'test':
            train_nodes_index = [i for i in range(gdata.node_num)]
            random.shuffle(train_nodes_index)
            temp_target_nodes_index = train_nodes_index[:len(gdata.target_nodes_index)]
            temp_shadow_nodes_index = train_nodes_index[len(gdata.target_nodes_index):len(gdata.target_nodes_index)+len(gdata.shadow_nodes_index)]
            temp_real_nodes_index = train_nodes_index[len(gdata.target_nodes_index)+len(gdata.shadow_nodes_index):len(gdata.target_nodes_index)+len(gdata.shadow_nodes_index)+len(gdata.attacker_nodes_index)]
            temp_test_nodes_index = train_nodes_index[len(gdata.target_nodes_index)+len(gdata.shadow_nodes_index)+len(gdata.attacker_nodes_index):]
            
            temp_gdata = copy.deepcopy(gdata)
            temp_gdata.target_nodes_index = temp_target_nodes_index
            temp_gdata.shadow_nodes_index = temp_shadow_nodes_index
            temp_gdata.attacker_nodes_index = temp_real_nodes_index
            temp_gdata.test_nodes_index = temp_test_nodes_index

            target_graph_data, _, _, _ = split_subgraph(temp_gdata)


        for epoch in range(args.benign_train_epochs):
            gnn_model.train()
            optimizer.zero_grad()
            input_data = target_graph_data.features.to(device), target_graph_data.adjacency.to(device)
            labels = target_graph_data.labels.to(device)
            _, output = gnn_model(input_data)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            train_correct_num = 0
            if (epoch + 1) % 50 == 0:
                _, output = gnn_model(input_data)
                predictions = predict_fn(output)
                
                for i in range(predictions.shape[0]):
                    if predictions[i, 0] == labels[i]:
                        train_correct_num += 1
                train_acc = train_correct_num / predictions.shape[0] * 100

                if last_train_acc == 0.0:
                    last_train_acc = train_acc
                else:
                    train_acc_diff = (train_acc - last_train_acc) / last_train_acc * 100
                    if train_acc_diff <= 0.5: #0.5%
                        break
                    else:
                        last_train_acc = train_acc
        
        torch.save(gnn_model, model_save_path)
    
    test_correct_num = 0
    gnn_model.eval()
    input_data = test_graph_data.features.to(device), test_graph_data.adjacency.to(device)
    _, output = gnn_model(input_data)
    predictions = predict_fn(output)
    test_labels = test_graph_data.labels
    for i in range(predictions.shape[0]):
        if predictions[i, 0] == test_labels[i]:
            test_correct_num += 1
    test_acc = test_correct_num / predictions.shape[0] * 100
    save_test_acc = round(test_acc, 3)

    graphs = [target_graph_data, shadow_graph_data, attacker_graph_data, test_graph_data]
    
    return graphs, gnn_model, save_test_acc


def get_possibility_variance(graph_data, gnn_model, measure_nodes):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    gnn_model.eval()
    gnn_model.to(device)

    input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    _, output = gnn_model(input_data)
    softmax = torch.nn.Softmax(dim=1)
    possibility = softmax(output)
    specific_possibility_vectors = possibility[measure_nodes].detach()
    variance = torch.var(specific_possibility_vectors, dim=1)

    return variance


def run(args, model_save_path, given_graph_data=None, process=None):

    if args.task_type == 'transductive':
        graph_data, gnn_model, test_acc = transductive_train(args, model_save_path, given_graph_data, process)
        return graph_data, gnn_model, test_acc
    elif args.task_type == 'inductive':
        graph_data, gnn_model, test_acc = inductive_train(args, model_save_path, given_graph_data, process) # multiple graph data
        return graph_data, gnn_model, test_acc


if __name__ == '__main__':
    pass
    