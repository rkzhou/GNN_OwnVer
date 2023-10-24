import random
import copy
import math
import utils.datareader
import benign
import extraction
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from statistics import mean


def insert_backdoor(args):
    data = utils.datareader.get_data(args)
    clean_data = utils.datareader.GraphData(data, args)
    bkd_data = copy.deepcopy(clean_data)

    # select nodes index for triggers
    train_node_num = math.ceil(args.backdoor_train_node_ratio * len(bkd_data.benign_train_nodes_index))
    temp_node_index = copy.deepcopy(bkd_data.benign_train_nodes_index)
    random.shuffle(temp_node_index)
    bkd_train_node_index = temp_node_index[:train_node_num]

    test_node_num = math.ceil(args.backdoor_test_node_ratio * len(bkd_data.test_nodes_index))
    temp_node_index = copy.deepcopy(bkd_data.test_nodes_index)
    random.shuffle(temp_node_index)
    bkd_test_node_index = temp_node_index[:test_node_num]

    # select feature index for triggers
    feature_index = [i for i in range(bkd_data.feat_dim)]
    random.shuffle(feature_index)
    feature_index = feature_index[:args.backdoor_feature_num]

    trigger_value = torch.rand(args.backdoor_feature_num)
    trigger_value = torch.where(trigger_value > 0.5, 1.0, 0.0)

    # generate triggers
    for i in bkd_train_node_index:
        bkd_data.labels[i] = args.backdoor_target_label
        bkd_data.features[i, feature_index] = trigger_value
    
    for i in bkd_test_node_index:
        bkd_data.labels[i] = args.backdoor_target_label
        bkd_data.features[i, feature_index] = trigger_value
    
    return clean_data, bkd_data, bkd_train_node_index, bkd_test_node_index


def train_bkd_model(args, clean_data, bkd_data, bkd_test_node_index):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    clean_test_node_index = copy.deepcopy(bkd_data.test_nodes_index)
    for i in bkd_test_node_index:
        clean_test_node_index.remove(i)

    _, benign_model, _ = benign.run(args, clean_data)
    # test_performance(args, bkd_data, benign_model, clean_test_node_index, bkd_test_node_index)

    bkd_model = copy.deepcopy(benign_model)
    loss_fn = F.cross_entropy
    optimizer = torch.optim.Adam(bkd_model.parameters(), lr=args.backdoor_lr, weight_decay=args.backdoor_weight_decay, betas=(0.5, 0.999))
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.backdoor_lr_decay_steps, gamma=0.1)

    print('Training backdoor model')
    last_bkd_acc = 0.0
    for epoch in tqdm(range(args.backdoor_train_epochs)):
        bkd_model.train()
        optimizer.zero_grad()
        input_data = bkd_data.features.to(device), bkd_data.adjacency.to(device)
        labels = bkd_data.labels.to(device)
        _, output = bkd_model(input_data)
        loss = loss_fn(output[bkd_data.benign_train_nodes_index], labels[bkd_data.benign_train_nodes_index])
        loss.backward()

        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            _, bkd_acc = test_performance(args, bkd_data, bkd_model, clean_test_node_index, bkd_test_node_index)
            if last_bkd_acc == 0.0:
                last_bkd_acc = bkd_acc
            else:
                bkd_acc_diff = bkd_acc - last_bkd_acc
                if bkd_acc_diff < 0.5:
                    break
                else:
                    last_bkd_acc = bkd_acc

    return bkd_model, clean_test_node_index


def test_performance(args, graph_data, model, clean_test_node_index, bkd_test_node_index):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.eval()
    predict_fn = lambda output: output.max(1, keepdim=True)[1]
    input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    labels = graph_data.labels.to(device)
    _, output = model(input_data)
    pred = predict_fn(output)

    clean_correct_num = 0
    bkd_correct_num = 0
    for i in clean_test_node_index:
        if pred[i, 0] == labels[i]:
            clean_correct_num += 1
    for i in bkd_test_node_index:
        if pred[i, 0] == args.backdoor_target_label:
            bkd_correct_num += 1
    clean_test_acc = clean_correct_num / len(clean_test_node_index) * 100
    bkd_test_acc = bkd_correct_num / len(bkd_test_node_index) * 100

    clean_test_acc = round(clean_test_acc, 3)
    bkd_test_acc = round(bkd_test_acc, 3)
    
    return clean_test_acc, bkd_test_acc


def get_stats_of_list(l, flag):
    mean_value = round(mean(l), 3)
    max_value = round(max(l), 3)
    min_value = round(min(l), 3)

    print(flag)
    print(mean_value, max_value, min_value)

    return mean_value, max_value, min_value


def run(args): 
    ind_clean_acc_list, ind_bkd_acc_list = list(), list()
    ext_clean_acc_list, ext_bkd_acc_list = list(), list()
    for _ in range(10):       
        clean_data, bkd_data, bkd_train_node_index, bkd_test_node_index = insert_backdoor(args)
        bkd_model, clean_test_node_index = train_bkd_model(args, clean_data, bkd_data, bkd_test_node_index)
        extraction_model, _, _ = extraction.run(args, clean_data, bkd_model)
                    
        ind_clean_acc, ind_bkd_acc = test_performance(args, bkd_data, bkd_model, clean_test_node_index, bkd_test_node_index)
        ind_clean_acc_list.append(ind_clean_acc)
        ind_bkd_acc_list.append(ind_bkd_acc)
        ext_clean_acc, ext_bkd_acc = test_performance(args, bkd_data, extraction_model, clean_test_node_index, bkd_test_node_index)
        ext_clean_acc_list.append(ext_clean_acc)
        ext_bkd_acc_list.append(ext_bkd_acc)
    
    get_stats_of_list(ind_clean_acc_list, 'original clean node accuracy:')
    get_stats_of_list(ind_bkd_acc_list, 'original backdoor node accuracy:')
    get_stats_of_list(ext_clean_acc_list, 'extraction clean node accuracy:')
    get_stats_of_list(ext_bkd_acc_list, 'extraction backdoor node accuracy:')
    


if __name__ == '__main__':
    pass