import torch
import benign
import extraction
import random
import math
import utils.datareader
import model.gnn_models
import model.extraction_models
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.mlp import mlp_nn
import boundary
from statistics import mean


def extract_logits(graph_data, specific_nodes, independent_model, surrogate_model):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    independent_model.eval()
    surrogate_model.eval()
    
    input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    _, independent_output = independent_model(input_data)
    _, surrogate_output = surrogate_model(input_data)

    softmax = torch.nn.Softmax(dim=1)
    independent_logits = softmax(independent_output)
    surrogate_logits = softmax(surrogate_output)

    if specific_nodes != None:
        independent_logits = independent_logits[specific_nodes].detach().cpu()
        surrogate_logits = surrogate_logits[specific_nodes].detach().cpu()
    
    logits = {'independent': independent_logits, 'surrogate': surrogate_logits}
    
    return logits


def measure_logits(logits):
    independent_logits = logits['independent']
    surrogate_logits = logits['surrogate']
    
    independent_var = torch.var(independent_logits, axis=1)
    surrogate_var = torch.var(surrogate_logits, axis=1)
    
    distance_pair = {'label_0': independent_var, 'label_1': surrogate_var}
    
    return distance_pair


def preprocess_data_flatten(distance_pairs:list):
    total_label0, total_label1 = list(), list()

    for pair_index in range(len(distance_pairs)):
        label0_distance = torch.flatten(distance_pairs[pair_index]['label_0']).view(1, -1)
        label1_distance = torch.flatten(distance_pairs[pair_index]['label_1']).view(1, -1)
        
        total_label0.append(label0_distance)
        total_label1.append(label1_distance)
    
    processed_data = {'label0': total_label0, 'label1': total_label1}

    return processed_data


def train_classifier(distance_pairs:list):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    processed_data = preprocess_data_flatten(distance_pairs)
    dataset = utils.datareader.VarianceData(processed_data['label0'], processed_data['label1'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    hidden_layers = [128, 64]
    model = mlp_nn(dataset.data.shape[1], hidden_layers)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch_num = 1000
    
    model.to(device)
    for epoch_index in tqdm(range(epoch_num)):
        model.train()
        for _, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        
        if (epoch_index + 1) % 100 == 0:
            model.eval()
            correct = 0
            for _, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predictions = torch.max(outputs.data, 1)
                correct += (predictions == labels).sum().item()

            acc = correct / len(dataset) * 100
            print(acc)

    
    return model
        
def owner_verify(graph_data, suspicious_model, verifier_model, measure_nodes):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # watermark_model.to(device)
    suspicious_model.to(device)
    # watermark_model.eval()
    suspicious_model.eval()
    
    input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    # _, watermark_output = watermark_model(input_data)
    _, suspicious_output = suspicious_model(input_data)

    softmax = torch.nn.Softmax(dim=1)
    # watermark_logits = softmax(watermark_output)
    suspicious_logits = softmax(suspicious_output)

    if measure_nodes != None:
        # watermark_logits = watermark_logits[measure_nodes].detach().cpu()
        suspicious_logits = suspicious_logits[measure_nodes].detach().cpu()

    # watermark_var = torch.var(watermark_logits, axis=1)
    suspicious_var = torch.var(suspicious_logits, axis=1)
    distance = suspicious_var
    distance = torch.flatten(distance).view(1, -1)

    verifier_model.to(device)
    verifier_model.eval()

    inputs = distance.to(device)
    outputs = verifier_model(inputs)
    _, predictions = torch.max(outputs.data, 1)
    
    return predictions


def batch_ownver(args):
    original_first_layer_dim = [525, 425]
    original_second_layer_dim = [375, 300, 225, 150, 75]

    shadow_first_layer_dim = [600, 550, 500, 450, 400]
    shadow_second_layer_dim = [300, 250, 200, 150, 100]
    
    ind_correct_num_list, ind_false_num_list = list(), list()
    ext_correct_num_list, ext_false_num_list = list(), list()

    original_acc_list = list()
    mask_acc_list = list()
    shadow_independent_acc_list = list()
    shadow_extraction_acc_list = list()
    shadow_extraction_fide_list = list()
    test_independent_acc_list = list()
    test_extraction_acc_list = list()
    test_extraction_fide_list = list()
    
    
    trial_index = 0
    for i in original_first_layer_dim:
        for j in original_second_layer_dim:
            print('starting trial {}'.format(trial_index))
            trial_index += 1
            original_layers = list()
            original_layers.append(i)
            original_layers.append(j)
            original_layers.sort(reverse=True)

            args.benign_model = 'gcn'
            args.benign_hidden_dim = original_layers
        
        
            original_graph_data, original_model, original_model_acc = benign.run(args)
            mask_graph_data, mask_nodes = boundary.mask_graph_data(args, original_graph_data, original_model)
            _, mask_model, mask_model_acc = benign.run(args, mask_graph_data)
            original_acc_list.append(original_model_acc)
            mask_acc_list.append(mask_model_acc)
        
            measure_nodes = []
            for each_class_nodes in mask_nodes:
                measure_nodes += each_class_nodes
        
        
            pair_list = list()
        
            # train shadow models
            args.extraction_model = 'gcnExtract'
            for p in shadow_first_layer_dim:
                for q in shadow_second_layer_dim:
                    shadow_layers = list()
                    shadow_layers.append(p)
                    shadow_layers.append(q)
                    shadow_layers.sort(reverse=True)

                    args.benign_hidden_dim = shadow_layers
                    args.extraction_hidden_dim = shadow_layers
                    _, independent_model, independent_acc = benign.run(args, original_graph_data)
                    extraction_model, extraction_acc, extraction_fide = extraction.run(args, original_graph_data, mask_model)

                    shadow_independent_acc_list.append(independent_acc)
                    shadow_extraction_acc_list.append(extraction_acc)
                    shadow_extraction_fide_list.append(extraction_fide)

                    logits = extract_logits(original_graph_data, measure_nodes, independent_model, extraction_model)
                    variance_pair = measure_logits(logits)
                    pair_list.append(variance_pair)
            classifier_model = train_classifier(pair_list)
            # save_path = '../temp_results/model_states/watermark_classifiers/exp_1/' + 'model' + str(trial_epoch) + '.pt'
            # torch.save(wm_classifier_model.state_dict(), save_path)
            sta0, sta1, sta2, sta3 = batch_unit_test(args, original_graph_data, mask_model, classifier_model, measure_nodes, test_independent_acc_list, test_extraction_acc_list, test_extraction_fide_list)
            ind_correct_num_list.append(sta0)
            ind_false_num_list.append(sta1)
            ext_correct_num_list.append(sta3)
            ext_false_num_list.append(sta2)

    TP, FN = sum(ind_correct_num_list), sum(ind_false_num_list) # True Positive, False Negative
    TN, FP = sum(ext_correct_num_list), sum(ext_false_num_list) # True Negative, False Positive
    print(TP, FN, TN, FP)
    
    accuracy = (TP + TN) / (TP + FN + TN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = (2 * precision * recall) / (precision + recall)

    accuracy = round(accuracy, 3)
    precision = round(precision, 3)
    recall = round(recall, 3)
    f1_score = round(f1_score, 3)
    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)
    print('f1_score:', f1_score)

    get_stats_of_list(original_acc_list, 'original accuracy:')
    get_stats_of_list(mask_acc_list, 'mask accuracy:')
    get_stats_of_list(shadow_independent_acc_list, 'shadow independent model accuracy:')
    get_stats_of_list(shadow_extraction_acc_list, 'shadow extraction model accuracy:')
    get_stats_of_list(shadow_extraction_fide_list, 'shadow extraction model fidelity:')
    get_stats_of_list(test_independent_acc_list, 'test independent model accuracy:')
    get_stats_of_list(test_extraction_acc_list, 'test extraction model accuracy:')
    get_stats_of_list(test_extraction_fide_list, 'test extraction model fidelity:')
        

def batch_unit_test(args, graph_data, mask_model, classifier_model, measure_nodes, independent_acc_list, extraction_acc_list, extraction_fide_list):
    independent_arch = ['gcn', 'gat', 'sage']
    extraction_arch = ['gcnExtract', 'gatExtract', 'sageExtract']
    first_layers_dim = [1280, 1140, 1000]
    second_layers_dim = [960, 830, 700]
    
    overall_ind_pred0_num, overall_ind_pred1_num = 0, 0
    overall_ext_pred0_num, overall_ext_pred1_num = 0, 0
    
    for i in independent_arch:
        for j in extraction_arch:
            args.benign_model = i
            args.extraction_model = j
            for p in first_layers_dim:
                for q in second_layers_dim:
                    test_model_layers = list()
                    test_model_layers.append(p)
                    test_model_layers.append(q)
                    test_model_layers.sort(reverse=True)

                    args.benign_hidden_dim = test_model_layers
                    args.extraction_hidden_dim = test_model_layers
                    _, test_independent_model, test_independent_acc = benign.run(args, graph_data)
                    test_extraction_model, test_extraction_acc, test_extraction_fide = extraction.run(args, graph_data, mask_model)

                    independent_acc_list.append(test_independent_acc)
                    extraction_acc_list.append(test_extraction_acc)
                    extraction_fide_list.append(test_extraction_fide)

                    ind_pred = owner_verify(graph_data, test_independent_model, classifier_model, measure_nodes)
                    ext_pred = owner_verify(graph_data, test_extraction_model, classifier_model, measure_nodes)

                    if ind_pred == 0:
                        overall_ind_pred0_num += 1
                    else:
                        overall_ind_pred1_num += 1
                    if ext_pred == 0:
                        overall_ext_pred0_num += 1
                    else:
                        overall_ext_pred1_num += 1
    
    return overall_ind_pred0_num, overall_ind_pred1_num, overall_ext_pred0_num, overall_ext_pred1_num


def get_stats_of_list(l, flag):
    mean_value = round(mean(l), 3)
    max_value = round(max(l), 3)
    min_value = round(min(l), 3)

    print(flag)
    print(mean_value, max_value, min_value)

    return mean_value, max_value, min_value


    
if __name__ == '__main__':
    pass