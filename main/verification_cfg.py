import json

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
import time
import os
import yaml
import itertools
from datetime import timedelta

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
        independent_logits = independent_logits[specific_nodes].detach()
        surrogate_logits = surrogate_logits[specific_nodes].detach()
    
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
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    hidden_layers = [128, 64]
    model = mlp_nn(dataset.data.shape[1], hidden_layers)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch_num = 1000
    
    model.to(device)
    for epoch_index in range(epoch_num):
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
            # print(acc)
            if acc == 100:
                break

    
    return model
        
def owner_verify(graph_data, suspicious_model, verifier_model, measure_nodes):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    suspicious_model.to(device)
    suspicious_model.eval()
    
    input_data = graph_data.features.to(device), graph_data.adjacency.to(device)
    _, suspicious_output = suspicious_model(input_data)

    softmax = torch.nn.Softmax(dim=1)
    suspicious_logits = softmax(suspicious_output)

    if measure_nodes != None:
        suspicious_logits = suspicious_logits[measure_nodes].detach()

    suspicious_var = torch.var(suspicious_logits, axis=1)
    distance = suspicious_var
    distance = torch.flatten(distance).view(1, -1)

    verifier_model.to(device)
    verifier_model.eval()

    inputs = distance.to(device)
    outputs = verifier_model(inputs)
    _, predictions = torch.max(outputs.data, 1)
    
    return predictions


def join_path(*save_path):
    original_model_save_root = os.path.join(*save_path)
    if not os.path.exists(original_model_save_root):
        os.makedirs(original_model_save_root)
    return original_model_save_root

def join_name(hidden_dims):
    str_dims = [str(n) for n in hidden_dims]
    return "_".join(str_dims)

def generate_combinations(layer_dims, num_hidden_layers):
    combinations = list(itertools.product(layer_dims, repeat=num_hidden_layers))

    # 将元组转换为列表，因为输出需要是列表的形式
    return [list(combination) for combination in combinations]



def random_generate_arch(layer_dims, num_hidden_layers, seed):

    # first generate all possible arches, then shuffle, sample
    def _generate_combinations(layer_dims, num_hidden_layer):
        combinations = list(itertools.product(layer_dims, repeat=num_hidden_layer))
        return [list(combination) for combination in combinations]

    all_hidden_dims = []
    for num_hidden_layer in num_hidden_layers:
        all_hidden_dims += _generate_combinations(layer_dims, num_hidden_layer)

    random.seed(seed)
    random.shuffle(all_hidden_dims)
    return all_hidden_dims


class GNNVerification():
    def __init__(self, args, global_cfg, train_setting_cfg, test_setting_cfg):
        self.global_cfg = global_cfg

        self.test_setting_cfg = test_setting_cfg
        self.train_setting_cfg = train_setting_cfg
        self.args = args
        self.train_save_root = os.path.join(global_cfg["train_save_root"], args.dataset, args.task_type)
        self.test_save_root = os.path.join(global_cfg["test_save_root"], args.dataset, args.task_type)
        # one experimental setting

        self.mask_model_save_name = "{}_{}".format(self.global_cfg["target_model"], join_name(self.global_cfg["target_hidden_dims"]))

    def train_original_model(self):
        # save original model
        original_model_save_root = join_path(self.train_save_root, 'original_models')
        original_model_save_path = os.path.join(original_model_save_root,
                                                "{}_{}.pt".format(args.benign_model, join_name(self.global_cfg["target_hidden_dims"])))
        return benign.run(args, original_model_save_path)

    def geneate_mask_model(self):

        # generate mask model
        mask_graph_data, mask_nodes = boundary.mask_graph_data(self.args, self.original_graph_data, self.original_model)
        mask_model_save_root = join_path(self.train_save_root, "mask_models", self.args.mask_feat_type,
                                         "{}_{}".format(self.args.mask_node_ratio, self.args.mask_feat_ratio))

        mask_model_save_path = os.path.join(mask_model_save_root, "{}.pt".format(self.mask_model_save_name))
        _, mask_model, mask_model_acc = benign.run(self.args, mask_model_save_path, mask_graph_data)

        measure_nodes = []
        for each_class_nodes in mask_nodes:
            measure_nodes += each_class_nodes

        return mask_model, mask_model_acc, measure_nodes

    # all model generate by this function will automaticly add a final layer for grove
    def train_models_by_arch(self, setting_cfg, model_arch, model_save_root, seed,
                             mask_model_save_name=None, mask_model=None, stage="train"):

        hidden_dims_generator = random_generate_arch(setting_cfg["layer_dims"], setting_cfg["num_hidden_layers"],
                                                     seed=seed)

        if len(hidden_dims_generator) < setting_cfg["num_model_per_arch"]:
            raise Exception("Can not generate enough unique model hidden dims, please reduce num_model_per_arch")

        model_list, acc_list, fidelity_list = [], [], []
        # generate num_model_per_arch models
        for hidden_dims, _ in zip(hidden_dims_generator, list(range(setting_cfg["num_model_per_arch"]))):

            # Important! add a fixed layer
            hidden_dims.append(self.global_cfg["embedding_dim"])
            if mask_model is None:
                # layer_dim, num_hidden_layers
                args.benign_hidden_dim = hidden_dims
                args.benign_model = model_arch
                # train independent model
                independent_model_save_root = join_path(model_save_root, 'independent_models')
                independent_model_save_path = os.path.join(independent_model_save_root,
                                                           "{}_{}_{}.pt".format(stage, args.benign_model,
                                                                                join_name(hidden_dims)))
                _, model, model_acc = benign.run(args, independent_model_save_path, self.original_graph_data)

            else:
                args.extraction_hidden_dim = hidden_dims
                args.extraction_model = model_arch
                extraction_model_save_root = join_path(model_save_root, 'extraction_models', args.mask_feat_type,
                                                       mask_model_save_name,
                                                       "{}_{}".format(args.mask_node_ratio, args.mask_feat_ratio))
                extraction_model_save_path = os.path.join(extraction_model_save_root,
                                                          "{}_{}_{}.pt".format(stage, args.extraction_model,
                                                                               join_name(hidden_dims)))
                model, model_acc, fidelity = extraction.run(args, extraction_model_save_path,
                                                            self.original_graph_data, mask_model, 'test')
                fidelity_list.append(fidelity)

            model_list.append(model)
            acc_list.append(model_acc)

        return model_list, acc_list, fidelity_list
    # This function train all models accroding to setting config
    def train_models_by_setting(self, setting_cfg,  model_save_root, mask_model_save_name=None, mask_model=None, stage="train"):
        all_model_list, all_acc_list, all_fidelity_list = [], [], []
        for seed, model_arch in enumerate(setting_cfg["model_arches"]):
            model_list, acc_list, fidelity_list = self.train_models_by_arch(setting_cfg, model_arch, model_save_root, seed, mask_model_save_name,
                                                                       mask_model=mask_model, stage=stage)
            all_model_list += model_list
            all_acc_list.append(acc_list)

            if mask_model is not None:
                all_fidelity_list.append(fidelity_list)

        return all_model_list, all_acc_list, all_fidelity_list

    def run_single_experiment(self):
        save_json = {}

        start = time.time()
        # train original model
        self.original_graph_data, self.original_model, self.original_model_acc = self.train_original_model()

        # generate mask model
        mask_start = time.time()
        self.mask_model, self.mask_model_acc, self.measure_nodes = self.geneate_mask_model()
        mask_run_time = time.time() - mask_start

        # train independent model
        train_inde_model_list, train_inde_acc_list, _ = self.train_models_by_setting(self.train_setting_cfg, self.train_save_root,
                                                                          mask_model=None, stage="train")
        # train surrogate model
        train_surr_model_list, train_surr_acc_list, train_surr_fidelity_list = self.train_models_by_setting(self.train_setting_cfg,
                                                                                                 self.train_save_root, self.mask_model_save_name,
                                                                                                            self.mask_model, stage="train")

        # extract
        pair_list = []
        for independent_model, extraction_model in zip(train_inde_model_list, train_surr_model_list):
            logits = extract_logits(self.original_graph_data, self.measure_nodes, independent_model, extraction_model)
            variance_pair = measure_logits(logits)
            pair_list.append(variance_pair)

        classifier_model = train_classifier(pair_list)

        # train independent  model
        test_inde_model_list, test_inde_acc_list, _ = self.train_models_by_setting(self.test_setting_cfg,  self.test_save_root,
                                                                      mask_model=None, stage="test")
        # train surrogate model
        test_surr_model_list, test_surr_acc_list, test_surr_fidelity_list = self.train_models_by_setting(
                                                                    self.test_setting_cfg, self.test_save_root,
                                                                    self.mask_model_save_name, self.mask_model, stage="test")

        TN, FP, FN, TP = 0, 0, 0, 0
        for test_independent_model, test_extraction_model in zip(test_inde_model_list, test_surr_model_list):
            ind_pred = owner_verify(self.original_graph_data, test_independent_model, classifier_model, self.measure_nodes)
            ext_pred = owner_verify(self.original_graph_data, test_extraction_model, classifier_model, self.measure_nodes)

            if ind_pred == 0:
                TN += 1
            else:
                FP += 1

            if ext_pred == 0:
                FN += 1
            else:
                TP += 1


        FPR = FP / (FP + TN)
        FNR = FN / (FN + TP)
        Accuracy = (TP + TN) / (TN + FP + TP + FN)

        # save to a
        save_json["TN"], save_json["TP"] = TN,  TP
        save_json["FN"], save_json["FP"] = FN, FP
        save_json["FPR"], save_json["FNR"] = FPR, FNR

        save_json["Accuracy"] = Accuracy
        save_json["original_model_acc"] = self.original_model_acc
        save_json["mask_model_acc"] = self.mask_model_acc
        save_json["train_inde_acc_list"] = train_inde_acc_list
        save_json["train_surr_acc_list"] = train_surr_acc_list
        save_json["train_surr_fidelity_list"] = train_surr_fidelity_list

        save_json["test_inde_acc_list"] = test_inde_acc_list
        save_json["test_surr_acc_list"] = test_surr_acc_list
        save_json["test_surr_fidelity_list"] = test_surr_fidelity_list

        json_save_root = join_path(self.global_cfg["res_path"], args.dataset, args.task_type, self.args.mask_feat_type,
                                               "{}_{}".format(args.mask_node_ratio, args.mask_feat_ratio))

        save_json["total_time"] = time.time()-start
        save_json["mask_run_time"] = mask_run_time
        with open("{}/{}_trainsetting{}_testsetting{}.json".format(json_save_root, self.mask_model_save_name,
                                                                   self.global_cfg["train_setting"],
                                                                    self.global_cfg["test_setting"]), "w") as f:
            f.write(json.dumps(save_json))

        return TP, FN, TN, FP

def multiple_experiments():

    config_path = "../config"
    with open(os.path.join(config_path, "global_cfg.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)

    # ['dblp', 'citeseer_full', 'pubmed', 'coauthor_phy', 'acm', 'amazon_photo']
    target_arch_list = ["gat", "gcn", "sage"]
    target_hidden_dim_list = [[352, 256],[288, 256],[224, 256]]
    attack_setting_list = [2]

    # load setting
    with open(os.path.join(config_path,'train_setting{}.yaml'.format(global_cfg["train_setting"])), 'r') as file:
        train_setting_cfg = yaml.safe_load(file)

    grid_params = []
    for dataset in [global_cfg["dataset"]]:
        for test_setting in attack_setting_list:
            for target_arch in target_arch_list:
                for target_hidden_dims in target_hidden_dim_list:
                    grid_params.append([dataset, test_setting, target_arch, target_hidden_dims])


    for dataset, test_setting, target_arch, target_hidden_dims in grid_params:
        args.dataset = dataset
        global_cfg['test_setting'] = test_setting
        global_cfg['target_model'] = target_arch
        global_cfg['target_hidden_dims'] = target_hidden_dims

        # load setting
        with open(os.path.join(config_path, 'test_setting{}.yaml'.format(test_setting)), 'r') as file:
            test_setting_cfg = yaml.safe_load(file)

        gnn_verification = GNNVerification(args, global_cfg, train_setting_cfg, test_setting_cfg)
        gnn_verification.run_single_experiment()

if __name__ == '__main__':
    from utils.config import parse_args
    args = parse_args()
    # ownver(args)
    multiple_experiments()