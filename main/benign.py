import sys, os
sys.path.append(os.path.abspath('..'))
import torch
import copy
import utils.config
import utils.datareader
import model.gcn
import model.graphsage
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from utils.config import parse_args


def normal_train(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    data = utils.datareader.get_data(args)
    gdata = utils.datareader.GraphData(data, args.benign_train_ratio)
    
    if args.benign_model == 'gcn':
        gnn_model = model.gcn.GCN(gdata.feat_dim, gdata.class_num, hidden_dim=args.benign_hidden_dim, dropout=args.benign_dropout)
    elif args.benign_model == 'sage':
        gnn_model = model.graphsage.GraphSage(gdata.feat_dim, gdata.class_num, hidden_dim=args.benign_hidden_dim, dropout=args.benign_dropout)
    gnn_model.to(device)

    # training
    loss_fn = F.cross_entropy
    predict_fn = lambda output: output.max(1, keepdim=True)[1]
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.benign_lr, weight_decay=args.benign_weight_decay, betas=(0.5, 0.999))
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.benign_lr_decay_steps, gamma=0.1)

    print('Training benign model')
    for epoch in tqdm(range(args.benign_train_epochs)):
        gnn_model.train()
        optimizer.zero_grad()
        input_data = gdata.features.to(device), gdata.adjacency.to(device)
        gdata.labels = gdata.labels.to(device)
        _, output = gnn_model(input_data)
        loss = loss_fn(output[gdata.train_mask], gdata.labels[gdata.train_mask])
        loss.backward()
        #print('training loss in epoch %d is %.4f' % (epoch, loss.item()))
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            gnn_model.eval()
            _, output = gnn_model(input_data)
            pred = predict_fn(output)
            test_pred = pred[gdata.test_mask]
            test_labels = gdata.labels[gdata.test_mask]
            correct_num = 0
            for i in range(test_pred.shape[0]):
                if test_pred[i, 0] == test_labels[i]:
                    correct_num += 1
            test_acc = correct_num / test_pred.shape[0] * 100
            print('Testing accuracy is %.4f' % (test_acc))
    
    return gdata, gnn_model


def antidistill_train(args, gnn_model, bkd_data, bkd_train_node_index, bkd_test_node_index):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    clean_train_node_index = copy.deepcopy(bkd_data.train_nodes_index)
    for i in bkd_train_node_index:
        clean_train_node_index.remove(i)

    clean_test_node_index = copy.deepcopy(bkd_data.test_nodes_index)
    for i in bkd_test_node_index:
        clean_test_node_index.remove(i)
    
    if gnn_model == None:
        if args.benign_model == 'gcn':
            gnn_model = model.gcn.GCN(bkd_data.feat_dim, bkd_data.class_num, hidden_dim=args.benign_hidden_dim, dropout=args.benign_dropout)
        elif args.benign_model == 'sage':
            gnn_model = model.graphsage.GraphSage(bkd_data.feat_dim, bkd_data.class_num, hidden_dim=args.benign_hidden_dim, dropout=args.benign_dropout)
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


def run(args, bkd_data=None):
    if args.benign_train_method == 'normal':
        graph_data, gnn_model = normal_train(args)
    elif args.benign_train_method == 'anti_distill':
        graph_data, gnn_model = antidistill_train(args, bkd_data)

    return graph_data, gnn_model


if __name__ == '__main__':
    pass
    