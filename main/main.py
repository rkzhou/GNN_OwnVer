import sys, os
sys.path.append(os.path.abspath('..'))
import benign
import extraction
import backdoor
import torch
import boundary
import model.graphsage
import model.gcn
import random
import grove
from utils.config import parse_args


def ownver(args):
    graph_data, benign_model = benign.run(args)
    # boundary.find_topk_feats_with_zerro(args, 40, graph_data, benign_model)

    watermarked_graph_data, watermarked_nodes = boundary.poison_graph_data(graph_data, 0, 0, benign_model)
    new_graph_data, watermarked_model = benign.run(args, watermarked_graph_data)

    extraction_emb_model, extraction_clf_model = extraction.run(args, graph_data, watermarked_model)

    boundary.measure_posteriors(graph_data, None, benign_model, None)
    boundary.measure_posteriors(graph_data, None, watermarked_model, None)
    boundary.measure_posteriors(graph_data, None, extraction_emb_model, extraction_clf_model)
    
    # bkd_data, bkd_model, bkd_train_node_index, bkd_test_node_index = backdoor.run(args, graph_data, benign_model)
    # extraction_emb_model, extraction_clf_model = extraction.run(args, graph_data, bkd_model)
    # backdoor.test_performance(args, bkd_data, extraction_emb_model, extraction_clf_model, bkd_test_node_index)


if __name__ == '__main__':
    args = parse_args()
    ownver(args)
    
    # graph_data, benign_model = benign.run(args)
    # watermarked_graph_data, watermarked_nodes = boundary.poison_graph_data(graph_data, 150, 10, benign_model)
    # clean_model, watermark_model, surrogate_model = grove.train_models(graph_data, watermarked_graph_data)
    # logits = grove.extract_logits(graph_data, clean_model, watermark_model, surrogate_model)
    # distance_pair = grove.measure_logits(graph_data, logits, [0,1,2,3,4,5,6,7,8,9,10])
    # print(distance_pair)