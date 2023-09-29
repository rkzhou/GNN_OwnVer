import sys, os
sys.path.append(os.path.abspath('..'))
import benign
import extraction
import boundary
import torch
import model.mlp
import verification
import random
import utils.graph_operator
import utils.evaluation
import time
from utils.config import parse_args

def ownver(args):
    benign_graph_data, benign_model = benign.run(args)

    watermark_graph_data, watermark_nodes = boundary.mask_graph_data(args, benign_graph_data, benign_model)

    #args.mask_feat_num = 0
    _, watermark_model = benign.run(args, watermark_graph_data)

    # # independent model for test
    # args.benign_model = 'sage'
    # args.benign_hidden_dim = [1024, 512]
    # _, suspicious_model = benign.run(args)

    # extraction model for test
    args.extraction_model = 'gcnExtract'
    args.extraction_hidden_dim = [64, 32]
    suspicious_model, _ = extraction.run(args, benign_graph_data, watermark_model)


    benign_posteriors = boundary.measure_posteriors(benign_graph_data, watermark_nodes, benign_model)
    watermark_posteriors = boundary.measure_posteriors(benign_graph_data, watermark_nodes, watermark_model)
    extraction_posteriors = boundary.measure_posteriors(benign_graph_data, watermark_nodes, suspicious_model)


def batch_ownver(args, trial_num, unit_test_num):
    independent_arch = ['sage']
    hidden_layers_num = [1, 2]
    model_layers = [24, 48, 96, 192, 384, 768]
    
    nwm_ind_correct_num_list, nwm_ind_false_num_list = list(), list()
    nwm_ext_correct_num_list, nwm_ext_false_num_list = list(), list()
    wm_ind_correct_num_list, wm_ind_false_num_list = list(), list()
    wm_ext_correct_num_list, wm_ext_false_num_list = list(), list()
    
    for trial_epoch in range(trial_num):
        print('starting trial {}'.format(trial_epoch))
        original_arch = random.choice(independent_arch)
        original_layers_num = random.choice(hidden_layers_num)
        original_layers = random.choices(model_layers, k=original_layers_num)
        original_layers.sort(reverse=True)

        args.benign_model = original_arch
        args.benign_hidden_dim = original_layers
        
        
        original_graph_data, original_model = benign.run(args)
        watermark_graph_data, watermark_nodes = boundary.mask_graph_data(args, original_graph_data, original_model)
        _, watermark_model = benign.run(args, watermark_graph_data)
        
        measure_nodes = []
        for each_class_nodes in watermark_nodes:
            measure_nodes += each_class_nodes
        
        
        nwm_pair_list, wm_pair_list = list(), list()
        # test with non-watermark model
        for group_index in range(args.verification_train_num):
            independent_model, extraction_model = verification.train_models(args, original_graph_data, original_model)
            logits = verification.extract_logits(original_graph_data, measure_nodes, independent_model, extraction_model)
            distance_pair = verification.measure_logits(logits)
            nwm_pair_list.append(distance_pair)
        nwm_classifier_model = verification.train_classifier(nwm_pair_list, 'flatten')
        # save_path = '../temp_results/model_states/nonwatermark_classifiers/exp_1/' + 'model' + str(trial_epoch) + '.pt'
        # torch.save(nwm_classifier_model.state_dict(), save_path)
        #the number of independent models predicted to 0, of independent models predicted to 1, of extraction models predicted to 0, of extraction models predicted to 1
        nwm_sta0, nwm_sta1, nwm_sta2, nwm_sta3 = batch_unit_test(args, original_graph_data, original_model, nwm_classifier_model, measure_nodes, unit_test_num, 'nonwatermark')
        nwm_ind_correct_num_list.append(nwm_sta0)
        nwm_ind_false_num_list.append(nwm_sta1)
        nwm_ext_correct_num_list.append(nwm_sta3)
        nwm_ext_false_num_list.append(nwm_sta2)
        
        # test with watermark model
        for group_index in range(args.verification_train_num):
            independent_model, extraction_model = verification.train_models(args, original_graph_data, watermark_model)
            logits = verification.extract_logits(original_graph_data, measure_nodes, independent_model, extraction_model)
            distance_pair = verification.measure_logits(logits)
            wm_pair_list.append(distance_pair)
        wm_classifier_model = verification.train_classifier(wm_pair_list, 'flatten')
        # save_path = '../temp_results/model_states/watermark_classifiers/exp_1/' + 'model' + str(trial_epoch) + '.pt'
        # torch.save(wm_classifier_model.state_dict(), save_path)
        wm_sta0, wm_sta1, wm_sta2, wm_sta3 = batch_unit_test(args, original_graph_data, watermark_model, wm_classifier_model, measure_nodes, unit_test_num, 'watermark')
        wm_ind_correct_num_list.append(wm_sta0)
        wm_ind_false_num_list.append(wm_sta1)
        wm_ext_correct_num_list.append(wm_sta3)
        wm_ext_false_num_list.append(wm_sta2)

    total_nwm_ind_correct_num, total_nwm_ind_false_num = sum(nwm_ind_correct_num_list), sum(nwm_ind_false_num_list)
    total_nwm_ext_correct_num, total_nwm_ext_false_num = sum(nwm_ext_correct_num_list), sum(nwm_ext_false_num_list)
    total_wm_ind_correct_num, total_wm_ind_false_num = sum(wm_ind_correct_num_list), sum(wm_ind_false_num_list)
    total_wm_ext_correct_num, total_wm_ext_false_num = sum(wm_ext_correct_num_list), sum(wm_ext_false_num_list)
    
    
    print(total_nwm_ind_correct_num, total_nwm_ind_false_num, total_nwm_ext_correct_num, total_nwm_ext_false_num)
    print(total_wm_ind_correct_num, total_wm_ind_false_num, total_wm_ext_correct_num, total_wm_ext_false_num)
    
    nwm_ind_correct_acc = total_nwm_ind_correct_num / (total_nwm_ind_correct_num + total_nwm_ind_false_num) * 100
    nwm_ext_correct_acc = total_nwm_ext_correct_num / (total_nwm_ext_correct_num + total_nwm_ext_false_num) * 100
    wm_ind_correct_acc = total_wm_ind_correct_num / (total_wm_ind_correct_num + total_wm_ind_false_num) * 100
    wm_ext_correct_acc = total_wm_ext_correct_num / (total_wm_ext_correct_num + total_wm_ext_false_num) * 100
 
    print('nonwatermark classifier independent models correct accuracy:', nwm_ind_correct_acc)
    print('nonwatermark classifier extraction models correct accuracy:', nwm_ext_correct_acc)
    print('watermark classifier independent correct accuracy:', wm_ind_correct_acc)
    print('watermark classifier extraction correct accuracy:', wm_ext_correct_acc)
        

def batch_unit_test(args, graph_data, watermark_model, classifier_model, measure_nodes, test_num, flag):
    independent_arch = ['gat', 'gcn']
    extraction_arch = ['gatExtract', 'gcnExtract']
    hidden_layers_num = [1, 2, 3]
    model_layers = [32, 64, 128, 256, 512, 1024]
    
    overall_ind_pred0_num, overall_ind_pred1_num = 0, 0
    overall_ext_pred0_num, overall_ext_pred1_num = 0, 0
    
    for trial_epoch in range(test_num):
        test_independent_arch = random.choice(independent_arch)
        test_independent_layers_num = random.choice(hidden_layers_num)
        test_independent_layers = random.choices(model_layers, k=test_independent_layers_num)
        test_independent_layers.sort(reverse=True)
        

        test_extraction_arch = random.choice(extraction_arch)
        test_extraction_layers_num = random.choice(hidden_layers_num)
        test_extraction_layers = random.choices(model_layers, k=test_extraction_layers_num)
        test_extraction_layers.sort(reverse=True)

        args.benign_model = test_independent_arch
        args.benign_hidden_dim = test_independent_layers
        args.extraction_model = test_extraction_arch
        args.extraction_hidden_dim = test_extraction_layers
            
        _, test_independent_model = benign.run(args, graph_data)
        test_extraction_model, _ = extraction.run(args, graph_data, watermark_model)
        
        ind_pred0_num, ind_pred1_num = 0, 0
        ext_pred0_num, ext_pred1_num = 0, 0
        for _ in range(5):
            ind_pred = verification.owner_verify(graph_data, test_independent_model, classifier_model, measure_nodes)
            ext_pred = verification.owner_verify(graph_data, test_extraction_model, classifier_model, measure_nodes)
            if ind_pred == 0:
                ind_pred0_num += 1
            else:
                ind_pred1_num += 1
            
            if ext_pred == 0:
                ext_pred0_num += 1
            else:
                ext_pred1_num += 1
        
        if ind_pred0_num >= ind_pred1_num:
            overall_ind_pred0_num += 1
        else:
            overall_ind_pred1_num += 1
        
        if ext_pred0_num >= ext_pred1_num:
            overall_ext_pred0_num += 1
        else:
            overall_ext_pred1_num += 1

    if flag == 'nonwatermark':
        print('nonwatermark test results')
        print(overall_ind_pred0_num, overall_ind_pred1_num)
        print(overall_ext_pred0_num, overall_ext_pred1_num)

    elif flag == 'watermark':
        print('watermark test results')
        print(overall_ind_pred0_num, overall_ind_pred1_num)
        print(overall_ext_pred0_num, overall_ext_pred1_num)
    
    return overall_ind_pred0_num, overall_ind_pred1_num, overall_ext_pred0_num, overall_ext_pred1_num


if __name__ == '__main__':
    args = parse_args()
    # ownver(args)
    batch_ownver(args, 10, 20)