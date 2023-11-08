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


def transductive_train(args, model_save_path, graph_data):
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
    if path.is_file():
        gnn_model = torch.load(model_save_path)
    else:
        if args.benign_model == 'gcn':
            gnn_model = model.gnn_models.GCN(gdata.feat_dim, gdata.class_num, hidden_dim=args.benign_hidden_dim)
        elif args.benign_model == 'sage':
            gnn_model = model.gnn_models.GraphSage(gdata.feat_dim, gdata.class_num, hidden_dim=args.benign_hidden_dim)
        elif args.benign_model == 'gat':
            gnn_model = model.gnn_models.GAT(gdata.feat_dim, gdata.class_num, hidden_dim=args.benign_hidden_dim)
        gnn_model.to(device)

        # training
        loss_fn = torch.nn.CrossEntropyLoss()
        predict_fn = lambda output: output.max(1, keepdim=True)[1]
        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.benign_lr)
        # scheduler = lr_scheduler.MultiStepLR(optimizer, args.benign_lr_decay_steps, gamma=0.1)

        last_train_acc = 0.0
        for epoch in range(args.benign_train_epochs):
            gnn_model.train()
            optimizer.zero_grad()
            input_data = gdata.features.to(device), gdata.adjacency.to(device)
            labels = gdata.labels.to(device)
            _, output = gnn_model(input_data)
            loss = loss_fn(output[gdata.benign_train_mask], labels[gdata.benign_train_mask])
            loss.backward()
            optimizer.step()
            # scheduler.step()

            train_correct_num = 0
            if (epoch + 1) % 100 == 0:
                _, output = gnn_model(input_data)
                pred = predict_fn(output)
                train_pred = pred[gdata.benign_train_mask]
                train_labels = gdata.labels[gdata.benign_train_mask]
                
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
    _, output = gnn_model(input_data)
    pred = predict_fn(output)
    test_pred = pred[gdata.test_mask]
    test_labels = gdata.labels[gdata.test_mask]
    for i in range(test_pred.shape[0]):
        if test_pred[i, 0] == test_labels[i]:
            test_correct_num += 1
    test_acc = test_correct_num / test_pred.shape[0] * 100
    save_test_acc = round(test_acc, 2)
    
    return gdata, gnn_model, save_test_acc


def inductive_train(args, model_save_path, graph_data):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if graph_data == None:
        data = utils.datareader.get_data(args)
        gdata = utils.datareader.GraphData(data, args)
        benign_train_greaph_data, extraction_train_graph_data, test_graph_data = split_subgraph(gdata)
    else:
        benign_train_greaph_data, extraction_train_graph_data, test_graph_data = graph_data

    path = Path(model_save_path)
    if path.is_file():
        gnn_model = torch.load(model_save_path)
    else:
        if args.benign_model == 'gcn':
            gnn_model = model.gnn_models.GCN(benign_train_greaph_data.feat_dim, benign_train_greaph_data.class_num, hidden_dim=args.benign_hidden_dim)
        elif args.benign_model == 'sage':
            gnn_model = model.gnn_models.GraphSage(benign_train_greaph_data.feat_dim, benign_train_greaph_data.class_num, hidden_dim=args.benign_hidden_dim)
        elif args.benign_model == 'gat':
            gnn_model = model.gnn_models.GAT(benign_train_greaph_data.feat_dim, benign_train_greaph_data.class_num, hidden_dim=args.benign_hidden_dim)
        gnn_model.to(device)

        loss_fn = torch.nn.CrossEntropyLoss()
        predict_fn = lambda output: output.max(1, keepdim=True)[1]
        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.benign_lr)

        last_train_acc = 0.0
        for epoch in range(args.benign_train_epochs):
            gnn_model.train()
            optimizer.zero_grad()
            input_data = benign_train_greaph_data.features.to(device), benign_train_greaph_data.adjacency.to(device)
            labels = benign_train_greaph_data.labels.to(device)
            _, output = gnn_model(input_data)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            train_correct_num = 0
            if (epoch + 1) % 100 == 0:
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
    save_test_acc = round(test_acc, 2)

    graphs = [benign_train_greaph_data, extraction_train_graph_data, test_graph_data]
    
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


def run(args, model_save_path, given_graph_data=None):
    if args.task_type == 'transductive':
        graph_data, gnn_model, test_acc = transductive_train(args, model_save_path, given_graph_data)
        return graph_data, gnn_model, test_acc
    elif args.task_type == 'inductive':
        graph_data, gnn_model, test_acc = inductive_train(args, model_save_path, given_graph_data) # multiple graph data
        return graph_data, gnn_model, test_acc


if __name__ == '__main__':
    pass
    