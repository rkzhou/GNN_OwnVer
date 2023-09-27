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


def train_models(args, graph_data, watermark_model):
    independent_arch = ['sage']
    extraction_arch = ['sageExtract']
    hidden_layers_num = [1, 2]
    model_layers = [256, 512, 1024, 2048]

    selected_independent_arch = random.choice(independent_arch)
    selected_independent_layers_num = random.choice(hidden_layers_num)
    selected_independent_layers = random.choices(model_layers, k=selected_independent_layers_num)
    selected_independent_layers.sort(reverse=True)

    selected_extraction_arch = random.choice(extraction_arch)
    selected_extraction_layers_num = random.choice(hidden_layers_num)
    selected_extraction_layers = random.choices(model_layers, k=selected_extraction_layers_num)
    selected_extraction_layers.sort(reverse=True)
    
    args.benign_model = selected_independent_arch
    args.benign_hidden_dim = selected_independent_layers
    args.extraction_model = selected_extraction_arch
    args.extraction_hidden_dim = selected_extraction_layers
    
    _, independent_model = benign.run(args, graph_data)
    extraction_model, _ = extraction.run(args, graph_data, watermark_model)
    
    return independent_model, extraction_model


def preprocess_data_grove(distance_pairs:list):
    total_label0, total_label1 = list(), list()

    for pair_index in range(len(distance_pairs)):
        label0_distance = distance_pairs[pair_index]['label_0']
        label1_distance = distance_pairs[pair_index]['label_1']

        total_label0.append(label0_distance)
        total_label1.append(label1_distance)

    processed_data = {'label0': total_label0, 'label1': total_label1}

    return processed_data


def preprocess_data_flatten(distance_pairs:list):
    total_label0, total_label1 = list(), list()

    for pair_index in range(len(distance_pairs)):
        label0_distance = torch.flatten(distance_pairs[pair_index]['label_0']).view(1, -1)
        label1_distance = torch.flatten(distance_pairs[pair_index]['label_1']).view(1, -1)
        
        total_label0.append(label0_distance)
        total_label1.append(label1_distance)
    
    processed_data = {'label0': total_label0, 'label1': total_label1}

    return processed_data


def train_classifier(distance_pairs:list, type):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    if type == 'grove':
        processed_data = preprocess_data_grove(distance_pairs)
    elif type == 'flatten':
        processed_data = preprocess_data_flatten(distance_pairs)
    dataset = utils.datareader.DistanceData(processed_data['label0'], processed_data['label1'])
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    hidden_layers = [128, 32]
    model = mlp_nn(dataset.data.shape[1], hidden_layers)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch_num = 1000
    
    model.to(device)
    for epoch_index in tqdm(range(epoch_num)):
        model.train()
        correct = 0
        for _, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        
            if (epoch_index+1) % 100 == 0:
                _, predictions = torch.max(outputs.data, 1)
                correct += (predictions == labels).sum().item()

        if (epoch_index+1) % 100 == 0:
            acc = correct / len(dataset) * 100
            print(acc)

    
    return model
        
def owner_verify(graph_data, watermark_model, suspicious_model, verifier_model, measure_nodes):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    watermark_model.to(device)
    suspicious_model.to(device)
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

    watermark_var = torch.var(watermark_logits, axis=1)
    suspicious_var = torch.var(suspicious_logits, axis=1)
    distance = watermark_var - suspicious_var
    distance = torch.flatten(distance).view(1, -1)

    verifier_model.to(device)
    verifier_model.eval()

    # predict_label0_num, predict_label1_num = 0, 0
    # inputs = distance.to(device)
    # outputs = verifier_model(inputs)
    # _, predictions = torch.max(outputs.data, 1)
    # for i in range(predictions.shape[0]):
    #     if predictions[i] == 0:
    #         predict_label0_num += 1
    #     elif predictions[i] == 1:
    #         predict_label1_num += 1
    
    # print('node number of label_0 is:', predict_label0_num)
    # print('node number of label_1 is:', predict_label1_num)

    inputs = distance.to(device)
    outputs = verifier_model(inputs)
    _, predictions = torch.max(outputs.data, 1)
    
    return predictions

    
if __name__ == '__main__':
    pass