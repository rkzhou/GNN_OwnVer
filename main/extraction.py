import torch
import random
import math
import copy
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import model.extraction_models
from tqdm import tqdm


class Classification(torch.nn.Module):

    def __init__(self, emb_size, num_classes):
        super(Classification, self).__init__()

        self.fc1 = torch.nn.Linear(emb_size, 256)
        self.fc2 = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


def evaluate_target_response(graph_data, model, response:str):
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

    if response == 'train_embeddings':
        target_response = embedding[graph_data.extraction_train_nodes_index]
    elif response == 'train_outputs':
        target_response = output[graph_data.extraction_train_nodes_index]
    elif response == 'test_embeddings':
        target_response = embedding[graph_data.test_nodes_index]
    elif response == 'test_outputs':
        target_response = output[graph_data.test_nodes_index]
        
    return target_response


def train_extraction_model(args, data):
    graph_data, train_emb, train_outputs, test_outputs = data
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #prepare model
    in_dim = graph_data.feat_dim
    if args.extraction_method == 'white_box':
        out_dim = train_emb.shape[1]
    elif args.extraction_method == 'black_box':
        out_dim = train_outputs.shape[1]

    if args.extraction_model == 'gcnExtract':
        extraction_model = model.extraction_models.GcnExtract(in_dim, out_dim, hidden_dim=args.extraction_hidden_dim)
    elif args.extraction_model == 'sageExtract':
        extraction_model = model.extraction_models.SageExtract(in_dim, out_dim, hidden_dim=args.extraction_hidden_dim)
    elif args.extraction_model == 'gatExtract':
        extraction_model = model.extraction_models.GatExtract(in_dim, out_dim, hidden_dim=args.extraction_hidden_dim)
    extraction_model = extraction_model.to(device)

    loss_fcn = torch.nn.MSELoss()
    loss_clf = torch.nn.CrossEntropyLoss()

    optimizer_embedding = torch.optim.Adam(extraction_model.parameters(), lr=args.extraction_lr)

    clf = None
    if args.extraction_method == 'white_box':
        clf = Classification(out_dim, graph_data.class_num)
        clf = clf.to(device)
        optimizer_classification = torch.optim.SGD(clf.parameters(), lr=args.extraction_lr)
    elif args.extraction_method == 'black_box':
        clf = None
    predict_fn = lambda output: output.max(1, keepdim=True)[1]
    
    last_train_acc, last_train_fide = 0.0, 0.0
    for epoch in tqdm(range(args.extraction_train_epochs)):
        extraction_model.train()
        #clf.train()
        train_emb = train_emb.to(device)
        train_outputs = train_outputs.to(device)
        input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
        extraction_embeddings, extraction_outputs = extraction_model(input_data)
        part_embeddings = extraction_embeddings[graph_data.extraction_train_nodes_index]
        part_outputs = extraction_outputs[graph_data.extraction_train_nodes_index]
        
        if args.extraction_method == 'white_box':
            optimizer_embedding.zero_grad()
            optimizer_classification.zero_grad()
            loss_emb = torch.sqrt(loss_fcn(part_embeddings, train_emb))
            loss_emb.backward()
            optimizer_embedding.step()

            outputs = clf(part_embeddings.detach())
            train_labels = predict_fn(train_outputs)
            train_labels = torch.flatten(train_labels)
            loss_out = loss_clf(outputs, train_labels)
            loss_out.backward()
            optimizer_classification.step()
        elif args.extraction_method == 'black_box':
            optimizer_embedding.zero_grad()
            if args.extraction_type == 'full':
                loss_emb = loss_fcn(part_outputs, train_outputs)
                loss_emb.backward()
            elif args.extraction_type == 'partial':
                distill_loss = loss_fcn(part_outputs, train_outputs)
                labels = graph_data.labels[graph_data.extraction_train_nodes_index]

                classify_loss = loss_clf(part_outputs, labels)
                total_loss = args.extraction_ratio * distill_loss + (1.0-args.extraction_ratio) * classify_loss
                total_loss.backward()
            optimizer_embedding.step()

        if (epoch + 1) % 50 == 0:
            extraction_model.eval()
            if args.extraction_method == 'white_box':
                clf.eval()

            acc_correct = 0
            fide_correct = 0

            embeddings, outputs = extraction_model(input_data)
            if args.extraction_method == 'white_box':
                outputs = clf(embeddings.detach())
            pred = predict_fn(outputs)
            test_labels = predict_fn(test_outputs)
            
            for i in range(len(graph_data.test_nodes_index)):
                if pred[graph_data.test_nodes_index[i]] == graph_data.labels[graph_data.test_nodes_index[i]]:
                    acc_correct += 1
                if pred[graph_data.test_nodes_index[i]] == test_labels[i]:
                    fide_correct += 1

            accuracy = acc_correct * 100.0 / len(graph_data.test_nodes_index)
            fidelity = fide_correct * 100.0 / test_outputs.shape[0]
            if last_train_acc == 0.0 or last_train_fide == 0.0:
                last_train_acc = accuracy
                last_train_fide = fidelity
            else:
                train_acc_diff = (accuracy - last_train_acc) / last_train_acc * 100
                train_fide_diff = (fidelity - last_train_fide) / last_train_fide * 100
                if train_acc_diff <= 0.5 and train_fide_diff <= 0.5:
                    break
                else:
                    last_train_acc = accuracy
                    last_train_fide = fidelity

    extraction_model.eval()
    if args.extraction_method == 'white_box':
        clf.eval()
    acc_correct, fide_correct = 0, 0
    input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    embeddings, outputs = extraction_model(input_data)
    if args.extraction_method == 'white_box':
        outputs = clf(embeddings.detach())
    pred = predict_fn(outputs)
    test_labels = predict_fn(test_outputs)
    for i in range(len(graph_data.test_nodes_index)):
        if pred[graph_data.test_nodes_index[i]] == graph_data.labels[graph_data.test_nodes_index[i]]:
            acc_correct += 1
        if pred[graph_data.test_nodes_index[i]] == test_labels[i]:
            fide_correct += 1
    accuracy = acc_correct * 100.0 / len(graph_data.test_nodes_index)
    fidelity = fide_correct * 100.0 / test_outputs.shape[0]
    print('Accuracy of model extraction is {:.4f} and fidelity is {:.4f}'.format(accuracy, fidelity))
    

    return extraction_model, clf


def run(args, graph_data, original_model):
    train_emb = evaluate_target_response(graph_data, original_model, 'train_embeddings') # we do not use this in black-box extraction setting
    train_outputs = evaluate_target_response(graph_data, original_model, 'train_outputs')
    test_outputs = evaluate_target_response(graph_data, original_model, 'test_outputs')
    extraction_data = graph_data, train_emb, train_outputs, test_outputs
    extraction_model = train_extraction_model(args, extraction_data)

    return extraction_model



if __name__ == '__main__':
    pass