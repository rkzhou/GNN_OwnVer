import sys, os
sys.path.append(os.path.abspath('..'))
import benign
import extraction
import backdoor
import torch
from utils.config import parse_args


def ownver(args):
    graph_data, benign_model = benign.run(args)
    bkd_data, bkd_model, bkd_test_node_index = backdoor.run(args, graph_data, benign_model)
    extraction_emb_model, extraction_clf_model = extraction.run(args, bkd_data, bkd_model)
    backdoor.test_performance(args, bkd_data, extraction_emb_model, extraction_clf_model, bkd_test_node_index)


if __name__ == '__main__':
    args = parse_args()
    ownver(args)