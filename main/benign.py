import torch
import copy
import utils.config
import utils.datareader
import model.gnn_models
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from utils.config import parse_args


def normal_train(args, graph_data):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if graph_data == None:
        data = utils.datareader.get_data(args)
        gdata = utils.datareader.GraphData(data, args)
    else:
        gdata = graph_data
    
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
    for epoch in tqdm(range(args.benign_train_epochs)):
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


def antidistill_train(args, gnn_model, bkd_data, bkd_train_node_index, bkd_test_node_index):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    clean_train_node_index = copy.deepcopy(bkd_data.benign_train_nodes_index)
    for i in bkd_train_node_index:
        clean_train_node_index.remove(i)

    clean_test_node_index = copy.deepcopy(bkd_data.test_nodes_index)
    for i in bkd_test_node_index:
        clean_test_node_index.remove(i)
    
    if gnn_model == None:
        if args.benign_model == 'gcn':
            gnn_model = model.gnn_models.GCN(bkd_data.feat_dim, bkd_data.class_num, hidden_dim=args.benign_hidden_dim, dropout=args.benign_dropout)
        elif args.benign_model == 'sage':
            gnn_model = model.gnn_models.GraphSage(bkd_data.feat_dim, bkd_data.class_num, hidden_dim=args.benign_hidden_dim, dropout=args.benign_dropout)
    gnn_model.to(device)

    #training
    loss_fn = F.cross_entropy
    predict_fn = lambda output: output.max(1, keepdim=True)[1]
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.benign_lr, weight_decay=args.benign_weight_decay, betas=(0.5, 0.999))
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.benign_lr_decay_steps, gamma=0.1)

    print('Training benign model')
    for epoch in tqdm(range(args.benign_train_epochs)):
        gnn_model.train()
        optimizer.zero_grad()
        input_data = bkd_data.features.to(device), bkd_data.adjacency.to(device)
        bkd_data.labels = bkd_data.labels.to(device)
        _, output = gnn_model(input_data)

        loss_clean = loss_fn(output[clean_train_node_index], bkd_data.labels[clean_train_node_index])
        loss_bkd = loss_fn(output[bkd_train_node_index], bkd_data.labels[bkd_train_node_index])
        total_loss = loss_clean + args.antidistill_train_ratio * loss_bkd
        total_loss.backward()
        #print('training loss in epoch %d is %.4f' % (epoch, loss.item()))
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            gnn_model.eval()
            _, output = gnn_model(input_data)
            pred = predict_fn(output)
            
            clean_test_pred = pred[clean_test_node_index]
            clean_test_labels = bkd_data.labels[clean_test_node_index]
            clean_correct_num = 0
            for i in range(len(clean_test_node_index)):
                if clean_test_pred[i] == clean_test_labels[i]:
                    clean_correct_num += 1
            clean_test_acc = clean_correct_num / len(clean_test_node_index) * 100
            print('Clean test accuracy is %.4f' % (clean_test_acc))

            bkd_test_pred = pred[bkd_test_node_index]
            bkd_test_labels = bkd_data.labels[bkd_test_node_index]
            bkd_correct_num = 0
            for i in range(len(bkd_test_node_index)):
                if bkd_test_pred[i] == bkd_test_labels[i]:
                    bkd_correct_num += 1
            bkd_test_acc = bkd_correct_num / len(bkd_test_node_index) * 100
            print('Backdoor test accuracy is %.4f' % (bkd_test_acc))
    
    return gnn_model


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
    specific_possibility_vectors = possibility[measure_nodes].detach().cpu()
    variance = torch.var(specific_possibility_vectors, dim=1)

    return variance


def run(args, given_graph_data=None, given_bkd_data=None):
    if args.benign_train_method == 'normal':
        graph_data, gnn_model, test_acc = normal_train(args, given_graph_data)
    elif args.benign_train_method == 'anti_distill':
        graph_data, gnn_model = antidistill_train(args, given_bkd_data)

    return graph_data, gnn_model, test_acc


if __name__ == '__main__':
    pass
    