import sys, os
sys.path.append(os.path.abspath('..'))

import torch
import random
import math
import copy
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from model.surrogate_sage import SageEmb


class Classification(torch.nn.Module):

    def __init__(self, emb_size, num_classes):
        super(Classification, self).__init__()

        self.fc1 = torch.nn.Linear(emb_size, num_classes)
        #self.fc2 = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        #x = F.relu(self.fc1(x))
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
    

def split_subset(node_index, split_ratio):
    temp_index = copy.deepcopy(node_index)
    random.shuffle(temp_index)
    subset_num = math.floor(len(node_index) * split_ratio)
    extraction_train_nodes_index = temp_index[:subset_num]
    extraction_test_nodes_index = temp_index[subset_num:]
    
    return extraction_train_nodes_index, extraction_test_nodes_index


def evaluate_target_response(model, graph_data, eval_nodes_index, response:str):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.eval()
    model = model.to(device)
    
    input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    embedding, output = model(input_data)
    embedding = embedding.detach()
    output = output.detach()

    if response == 'embeddings':
        target_response = embedding[eval_nodes_index]
    elif response == 'outputs':
        target_response = output[eval_nodes_index]
        
    return target_response


def train_surrogate_model(args, data):
    graph_data, surrogate_train_index, surrogate_test_index, train_emb, train_outputs, test_outputs = data
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #prepare model
    in_dim = graph_data.feat_dim
    out_dim = train_emb.shape[1]
    

    surrogate_model = SageEmb(in_dim, train_emb.shape[1], hidden_dim=args.extraction_hidden_dim, dropout=args.extraction_dropout)
    surrogate_model = surrogate_model.to(device)

    loss_fcn = torch.nn.MSELoss()
    loss_clf = torch.nn.MSELoss()

    optimizer_embbeding = torch.optim.Adam(surrogate_model.parameters(), lr=args.extraction_lr, weight_decay=args.extraction_weight_decay, betas=(0.5, 0.999))

    clf = Classification(out_dim, graph_data.class_num)
    clf = clf.to(device)
    predict_fn = lambda output: output.max(1, keepdim=True)[1]
    optimizer_classification = torch.optim.SGD(clf.parameters(), lr=args.extraction_lr)
    scheduler_classification = lr_scheduler.MultiStepLR(optimizer_classification, args.extraction_lr_decay_steps, gamma=0.1)

    print('Model Extracting')
    for epoch in tqdm(range(args.extraction_train_epochs)):
        surrogate_model.train()
        clf.train()

        train_emb = train_emb.to(device)
        train_outputs = train_outputs.to(device)
        optimizer_embbeding.zero_grad()
        input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
        surrogate_embeddings = surrogate_model(input_data)
        part_embeddings = surrogate_embeddings[surrogate_train_index]
        loss_emb = loss_fcn(part_embeddings, train_emb)
        loss_emb.backward()
        optimizer_embbeding.step()
        scheduler_classification.step()

        optimizer_classification.zero_grad()
        logists = clf(part_embeddings.detach())

        loss_pred = loss_clf(logists, train_outputs)
        loss_pred.backward()
        optimizer_classification.step()

        if (epoch + 1) % 100 == 0:
            surrogate_model.eval()
            clf.eval()

            acc_correct = 0
            fide_correct = 0

            embeddings = surrogate_model(input_data)
            outputs = clf(embeddings.detach())
            pred = predict_fn(outputs)
            test_labels = predict_fn(test_outputs)
            
            for i in range(len(surrogate_test_index)):
                if pred[surrogate_test_index[i]] == graph_data.labels[surrogate_test_index[i]]:
                    acc_correct += 1
                if pred[surrogate_test_index[i]] == test_labels[i]:
                    fide_correct += 1

            accuracy = acc_correct * 100.0 / len(surrogate_test_index)
            fidelity = fide_correct * 100.0 / test_outputs.shape[0]
            print('Accuracy of model extraction is {:.4f} and fidelity is {:.4f}'.format(accuracy, fidelity))
    

    return surrogate_model, clf


def run(args, graph_data, original_model):
    surrogate_train_index, surrogate_test_index = split_subset(graph_data.test_nodes_index, args.extraction_train_ratio)
    train_emb = evaluate_target_response(original_model, graph_data, surrogate_train_index, 'embeddings')
    train_outputs = evaluate_target_response(original_model, graph_data, surrogate_train_index, 'outputs')
    test_outputs = evaluate_target_response(original_model, graph_data, surrogate_test_index, 'outputs')
    extraction_data = graph_data, surrogate_train_index, surrogate_test_index, train_emb, train_outputs, test_outputs
    surrogate_model = train_surrogate_model(args, extraction_data)

    return surrogate_model



if __name__ == '__main__':
    pass