import sys, os
sys.path.append(os.path.abspath('..'))
import torch
import utils.config
import utils.datareader
import model.gcn
import model.graphsage
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from utils.config import parse_args


def run(args):
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
    


if __name__ == '__main__':
    args = parse_args()
    run(args)
    