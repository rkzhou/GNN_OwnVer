import random
import copy
import math
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm


def node_level_trigger(args, benign_data):
    trigger_value = random.random()
    bkd_data = copy.deepcopy(benign_data)

    # select nodes index for triggers
    train_node_num = math.ceil(args.backdoor_train_node_ratio * len(bkd_data.train_nodes_index))
    temp_node_index = copy.deepcopy(bkd_data.train_nodes_index)
    random.shuffle(temp_node_index)
    train_node_index = temp_node_index[:train_node_num]

    test_node_num = math.ceil(args.backdoor_test_node_ratio * len(bkd_data.test_nodes_index))
    temp_node_index = copy.deepcopy(bkd_data.test_nodes_index)
    random.shuffle(temp_node_index)
    test_node_index = temp_node_index[:test_node_num]

    # select feature index for triggers
    feature_index = [i for i in range(bkd_data.feat_dim)]
    random.shuffle(feature_index)
    feature_index = feature_index[:args.backdoor_feature_num]

    # generate triggers
    for i in train_node_index:
        bkd_data.labels[i] = args.backdoor_target_label
        for j in feature_index:
            bkd_data.features[i, j] = random.random()
    
    for i in test_node_index:
        bkd_data.labels[i] = args.backdoor_target_label
        for j in feature_index:
            bkd_data.features[i, j] = random.random()
    
    return bkd_data, train_node_index, test_node_index


def train_bkd_model(args, bkd_data, test_node_index, benign_model):
    clean_test_node_index = copy.deepcopy(bkd_data.test_nodes_index)
    for i in test_node_index:
        clean_test_node_index.remove(i)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    bkd_model = copy.deepcopy(benign_model)
    loss_fn = F.cross_entropy
    predict_fn = lambda output: output.max(1, keepdim=True)[1]
    optimizer = torch.optim.Adam(bkd_model.parameters(), lr=args.backdoor_lr, weight_decay=args.backdoor_weight_decay, betas=(0.5, 0.999))
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.backdoor_lr_decay_steps, gamma=0.1)

    print('Training backdoor model')
    for epoch in tqdm(range(args.backdoor_train_epochs)):
        bkd_model.train()
        optimizer.zero_grad()
        input_data = bkd_data.features.to(device), bkd_data.adjacency.to(device)
        bkd_data.labels = bkd_data.labels.to(device)
        _, output = bkd_model(input_data)
        loss = loss_fn(output[bkd_data.train_mask], bkd_data.labels[bkd_data.train_mask])
        loss.backward()
        #print('training loss in epoch %d is %.4f' % (epoch, loss.item()))
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            bkd_model.eval()
            _, output = bkd_model(input_data)
            pred = predict_fn(output)
            test_pred = pred
            test_labels = bkd_data.labels

            clean_correct_num = 0
            bkd_correct_num = 0
            for i in range(test_pred.shape[0]):
                if i in clean_test_node_index:
                    if test_pred[i, 0] == test_labels[i]:
                        clean_correct_num += 1
                elif i in test_node_index:
                    if test_pred[i, 0] == args.backdoor_target_label:
                        bkd_correct_num += 1
            clean_test_acc = clean_correct_num / len(clean_test_node_index) * 100
            bkd_test_acc = bkd_correct_num / len(test_node_index) * 100

            print('Clean testing accuracy is %.4f' % (clean_test_acc))
            print('Backdoor testing accuracy is %.4f' % (bkd_test_acc))

    return bkd_model


def test_performance(args, bkd_data, emb_model, clf_model, bkd_test_node_index):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    clean_test_node_index = copy.deepcopy(bkd_data.test_nodes_index)
    for i in bkd_test_node_index:
        clean_test_node_index.remove(i)

    emb_model.eval()
    #clf_model.eval()
    predict_fn = lambda output: output.max(1, keepdim=True)[1]
    input_data = bkd_data.features.to(device), bkd_data.adjacency.to(device)
    embeddings, outputs = emb_model(input_data)
    #outputs = clf_model(embeddings.detach())
    pred = predict_fn(outputs)

    clean_correct_num, bkd_correct_num = 0, 0
    for i in range(pred.shape[0]):
        if i in clean_test_node_index:
            if pred[i, 0] == bkd_data.labels[i]:
                clean_correct_num += 1
        elif i in bkd_test_node_index:
            if pred[i, 0] == args.backdoor_target_label:
                bkd_correct_num += 1
    clean_test_acc = clean_correct_num / len(clean_test_node_index) * 100
    bkd_test_acc = bkd_correct_num / len(bkd_test_node_index) * 100
    print('For extraction model, clean testing accuracy is %.4f and backdoor accuracy is %.4f' % (clean_test_acc, bkd_test_acc))


def run(args, benign_data, benign_model):
    bkd_data, bkd_train_node_index, bkd_test_node_index = node_level_trigger(args, benign_data)
    bkd_model = train_bkd_model(args, bkd_data, bkd_test_node_index, benign_model)
    
    return bkd_data, bkd_model, bkd_train_node_index, bkd_test_node_index


if __name__ == '__main__':
    pass