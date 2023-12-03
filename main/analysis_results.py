import os
import json
import statistics
import itertools
import numpy as np
# this file is used to process the results

def calculate_metrics(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1
def save_mean_std(res_list, field, log_f):
    metric = [res[field] for res in res_list]
    metric_mean, metric_std = statistics.mean(metric), statistics.stdev(metric)
    print("{}:{}+{}".format(field, round(metric_mean, 2), round(metric_std, 2)), file=log_f, end="\n")

def merge_results_setting(setting_path, arch="all"):

    res_list = []
    for file_name in os.listdir(setting_path):

        if not file_name.endswith("json"):
            continue

        if arch in ["gat", "gcn", "sage"] and (not file_name.startswith(arch)):
            continue

        with open(os.path.join(setting_path, file_name), "r") as file:
            res = json.loads(file.read())
            res_list.append(res)


    precision, recall, f1 = [], [], []
    for res in res_list:
        _precision, _recall, _f1 = calculate_metrics(res["TP"], res["FP"], res["FN"])
        precision.append(_precision)
        recall.append(_recall)
        f1.append(_f1)

    log_f = open(os.path.join(setting_path, "{}_res.txt".format(arch)), 'w')

    # , "mask_model_acc"
    for field in ["FPR", "FNR", "Accuracy", "original_model_acc", "mask_model_acc"]:
        save_mean_std(res_list, field, log_f)

       # flatten the list
    test_inde_acc = [ res["test_inde_acc_list"] for res in res_list]
    test_inde_acc = np.array(test_inde_acc).flatten().tolist()


    test_surr_acc = [ res["test_surr_acc_list"] for res in res_list]
    test_surr_acc = np.array(test_surr_acc).flatten().tolist()

    test_surr_fidelity = [ res["test_surr_fidelity_list"] for res in res_list]
    test_surr_fidelity = np.array(test_surr_fidelity).flatten().tolist()


    precision_mean, precision_std = statistics.mean(precision), statistics.stdev(precision)
    recall_mean, recall_std = statistics.mean(recall), statistics.stdev(recall)
    f1_mean, f1_std = statistics.mean(f1), statistics.stdev(f1)

    test_inde_acc_mean, test_inde_acc_std = statistics.mean(test_inde_acc), statistics.stdev(test_inde_acc)
    test_surr_acc_mean, test_surr_acc_std = statistics.mean(test_surr_acc), statistics.stdev(test_surr_acc)
    test_surr_fidelity_mean, test_surr_fidelity_std = statistics.mean(test_surr_fidelity), statistics.stdev(test_surr_fidelity)

    print("precision:{}+{}".format(round(precision_mean, 2), round(precision_std, 2)), file=log_f, end="\n")
    print("recall:{}+{}".format(round(recall_mean, 2), round(recall_std, 2)), file=log_f, end="\n")
    print("f1:{}+{}".format(round(f1_mean, 2), round(f1_std, 2)), file=log_f, end="\n")
    print("test_inde_acc_mean:{}+{}".format(round(test_inde_acc_mean, 2), round(test_inde_acc_std, 2)), file=log_f, end="\n")
    print("test_surr_acc_mean:{}+{}".format(round(test_surr_acc_mean, 2), round(test_surr_acc_std, 2)), file=log_f, end="\n")
    print("test_surr_fidelity_mean:{}+{}".format(round(test_surr_fidelity_mean, 2), round(test_surr_fidelity_std, 2)), file=log_f, end="\n")

    log_f.close()


if __name__ == '__main__':
    for node_ratio in [1.0]:
        for feat_ratio in [0.2]:
            for setting in [1, 2, 3, 4]:
                merge_results_setting('/home/ruikai/GNN_OwnVer/res/Citeseer/transductive/random_mask/{}_{}/train_setting1/test_setting{}'.format(node_ratio, feat_ratio, setting))