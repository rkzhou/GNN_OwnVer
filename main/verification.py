import torch
import extraction
import random
import math
import utils.datareader
import model.gnn_models
import model.extraction_models
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.mlp import mlp_nn


def extract_logits(graph_data, specific_nodes, independent_model, watermark_model, surrogate_model):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    independent_model.eval()
    watermark_model.eval()
    surrogate_model.eval()
    
    input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    _, independent_output = independent_model(input_data)
    _, watermark_output = watermark_model(input_data)
    _, surrogate_output = surrogate_model(input_data)

    softmax = torch.nn.Softmax(dim=1)
    independent_logits = softmax(independent_output)
    watermark_logits = softmax(watermark_output)
    surrogate_logits = softmax(surrogate_output)

    if specific_nodes != None:
        independent_logits = independent_logits[specific_nodes].detach().cpu()
        watermark_logits = watermark_logits[specific_nodes].detach().cpu()
        surrogate_logits = surrogate_logits[specific_nodes].detach().cpu()
    
    logits = {'independent': independent_logits, 'watermark': watermark_logits, 'surrogate': surrogate_logits}
    
    return logits


def measure_logits(logits):
    independent_logits = logits['independent']
    watermark_logits = logits['watermark']
    surrogate_logits = logits['surrogate']
    
    wi_distance = (watermark_logits - independent_logits).pow(2) # this distance should be far, label=0
    ws_distance = (watermark_logits - surrogate_logits).pow(2) # this distance should be close, label=1
    
    distance_pair = {'label_0': wi_distance, 'label_1': ws_distance}
    
    return distance_pair


def train_models(clean_data, watermark_model):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    input_dim, output_dim = clean_data.feat_dim, clean_data.class_num
    independent_arch = ['gcn', 'sage', 'gat']
    extraction_arch = ['gcnExtract', 'gatExtract', 'sageExtract']
    model_layers = [256, 128, 64, 32]

    selected_independent_arch = random.choice(independent_arch)
    selected_independent_layers = random.choices(model_layers, k=2)
    selected_independent_layers.sort(reverse=True)
    selected_extraction_arch = random.choice(extraction_arch)
    selected_extraction_layers = random.choices(model_layers, k=2)
    selected_extraction_layers.sort(reverse=True)

    if selected_independent_arch == 'gcn':
        independent_model = model.gnn_models.GCN(input_dim, output_dim, selected_independent_layers)
    elif selected_independent_arch == 'sage':
        independent_model = model.gnn_models.GraphSage(input_dim, output_dim, selected_independent_layers)
    elif selected_independent_arch == 'gat':
        independent_model = model.gnn_models.GAT(input_dim, output_dim, selected_independent_layers)

    if selected_extraction_arch == 'gcnExtract':
        extraction_model = model.extraction_models.GcnExtract(input_dim, output_dim, selected_extraction_layers)
    elif selected_extraction_arch == 'sageExtract':
        extraction_model = model.extraction_models.SageExtract(input_dim, output_dim, selected_extraction_layers)
    elif selected_extraction_arch == 'gatExtract':
        extraction_model = model.extraction_models.GatExtract(input_dim, output_dim, selected_extraction_layers)
    
    independent_optimizer = torch.optim.Adam(independent_model.parameters(), lr=0.001)
    extraction_optimizer = torch.optim.Adam(extraction_model.parameters(), lr=0.001)
    
    epoch_num = 1000
    normal_loss_fn = torch.nn.CrossEntropyLoss()
    extraction_loss_fn = torch.nn.MSELoss()
    predict_fn = lambda output: output.max(1, keepdim=True)[1]
    
    # use the original dataset to train the clean model
    independent_model.to(device)
    for epoch in tqdm(range(epoch_num)):
        independent_model.train()
        independent_optimizer.zero_grad()
        clean_input_data = clean_data.features.to(device), clean_data.adjacency.to(device)
        clean_labels = clean_data.labels.to(device)
        _, output = independent_model(clean_input_data)
        loss = normal_loss_fn(output[clean_data.benign_train_mask], clean_labels[clean_data.benign_train_mask])
        loss.backward()
        independent_optimizer.step()

        if (epoch + 1) % 100 == 0:
            independent_model.eval()
            _, output = independent_model(clean_input_data)
            pred = predict_fn(output)
            test_pred = pred[clean_data.test_mask]
            test_labels = clean_data.labels[clean_data.test_mask]
            correct_num = 0
            for i in range(test_pred.shape[0]):
                if test_pred[i, 0] == test_labels[i]:
                    correct_num += 1
            test_acc = correct_num / test_pred.shape[0] * 100
            print('Testing accuracy is %.4f' % (test_acc))
    
    extraction_train_outputs = extraction.evaluate_target_response(clean_data, watermark_model, 'train_outputs')
    extraction_test_outputs = extraction.evaluate_target_response(clean_data, watermark_model, 'test_outputs')
    extraction_model.to(device)
    for epoch in tqdm(range(epoch_num)):
        extraction_model.train()
        extraction_optimizer.zero_grad()
        extraction_train_outputs = extraction_train_outputs.to(device)
        extraction_input_data = clean_data.features.to(device), clean_data.adjacency.to(device)
        extraction_embeddings, extraction_outputs = extraction_model(extraction_input_data)
        part_outputs = extraction_outputs[clean_data.extraction_train_nodes_index]
        loss = extraction_loss_fn(part_outputs, extraction_train_outputs)
        loss.backward()
        extraction_optimizer.step()
        if (epoch + 1) % 100 == 0:
            extraction_model.eval()
            acc_correct = 0
            fide_correct = 0
            _, outputs = extraction_model(extraction_input_data)
            pred = predict_fn(outputs)
            test_labels = predict_fn(extraction_test_outputs)
            for i in range(len(clean_data.test_nodes_index)):
                if pred[clean_data.test_nodes_index[i]] == clean_data.labels[clean_data.test_nodes_index[i]]:
                    acc_correct += 1
                if pred[clean_data.test_nodes_index[i]] == test_labels[i]:
                    fide_correct += 1
            accuracy = acc_correct * 100.0 / len(clean_data.test_nodes_index)
            fidelity = fide_correct * 100.0 / extraction_test_outputs.shape[0]
            print('Accuracy of model extraction is {:.4f} and fidelity is {:.4f}'.format(accuracy, fidelity))
    
    return independent_model, extraction_model


def preprocess_data(distance_pairs:list, split_ratio):
    node_num, class_num = distance_pairs[0]['label_0'].shape
    train_node_num = math.floor(node_num * split_ratio)
    test_node_num = node_num - train_node_num

    total_label0_train, total_label0_test = list(), list()
    total_label1_train, total_label1_test = list(), list()

    for pair_index in range(len(distance_pairs)):
        node_mask = [i for i in range(node_num)]
        random.shuffle(node_mask)
        train_node_index = node_mask[:train_node_num]
        test_node_index = node_mask[train_node_num:]

        label0_distance = distance_pairs[pair_index]['label_0']
        label1_distance = distance_pairs[pair_index]['label_1']
        label0_train = label0_distance[train_node_index]
        label0_test = label0_distance[test_node_index]
        label1_train = label1_distance[train_node_index]
        label1_test = label1_distance[test_node_index]

        total_label0_train.append(label0_train)
        total_label0_test.append(label0_test)
        total_label1_train.append(label1_train)
        total_label1_test.append(label1_test)

    processed_data = {'label0_train': total_label0_train, 'label0_test': total_label0_test, 'label1_train': total_label1_train, 'label1_test': total_label1_test}

    return processed_data


def train_classifier(distance_pairs:list, split_ratio):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    processed_data = preprocess_data(distance_pairs, split_ratio)
    train_dataset = utils.datareader.DistanceData(processed_data['label0_train'], processed_data['label1_train'])
    test_dataset = utils.datareader.DistanceData(processed_data['label0_test'], processed_data['label1_test'])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    hidden_layers = [256, 128, 64]
    model = mlp_nn(train_dataset.data.shape[1], hidden_layers)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch_num = 500
    
    model.to(device)
    for epoch_index in tqdm(range(epoch_num)):
        model.train()
        for _, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        if (epoch_index+1) % 100 == 0:
            model.eval()
            correct = 0
            for _, (inputs, labels) in enumerate(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predictions = torch.max(outputs.data, 1)
                correct += (predictions == labels).sum().item()
            acc = correct / len(test_dataset) * 100
            print(acc)

    
    return model
        
def owner_verify(graph_data, watermark_model, suspicious_model, verifier_model, measure_nodes):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    watermark_model.eval()
    suspicious_model.eval()
    
    input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    _, watermark_output = watermark_model(input_data)
    _, suspicious_output = suspicious_model(input_data)

    softmax = torch.nn.Softmax(dim=1)
    watermark_logits = softmax(watermark_output)
    suspicious_logits = softmax(suspicious_output)

    if measure_nodes != None:
        watermark_logits = watermark_logits[measure_nodes].detach().cpu()
        suspicious_logits = suspicious_logits[measure_nodes].detach().cpu()
    
    distance = (watermark_logits - suspicious_logits).pow(2)

    verifier_model.to(device)
    verifier_model.eval()
    predict_label0_num, predict_label1_num = 0, 0
    inputs = distance.to(device)
    outputs = verifier_model(inputs)
    _, predictions = torch.max(outputs.data, 1)
    for i in range(predictions.shape[0]):
        if predictions[i] == 0:
            predict_label0_num += 1
        elif predictions[i] == 1:
            predict_label1_num += 1
    
    print('node number of label_0 is:', predict_label0_num)
    print('node number of label_1 is:', predict_label1_num)
    
    
if __name__ == '__main__':
    pass