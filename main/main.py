import sys, os
sys.path.append(os.path.abspath('..'))
import benign
import extraction
import boundary
import torch
import verification
import random
import utils.evaluation
import backdoor
import grove
from utils.config import parse_args


def ownver(args):
    benign_graph_data, benign_model, ori_acc = benign.run(args)

    mask_graph_data, mask_nodes = boundary.mask_graph_data(args, benign_graph_data[0], benign_model)

    graphs_data = [mask_graph_data, benign_graph_data[1], benign_graph_data[2]]

    _, mask_model, mask_acc = benign.run(args, graphs_data)


    # independent model for test
    args.benign_model = 'gcn'
    args.benign_hidden_dim = [64, 32]
    _, independent_model, ind_acc = benign.run(args, benign_graph_data)

    # extraction model for test
    args.extraction_model = 'gcnExtract'
    args.extraction_hidden_dim = [512, 256]
    extraction_model, ext_acc, ext_fide = extraction.run(args, benign_graph_data, mask_model)
    print(ori_acc, mask_acc)
    print(ind_acc, ext_acc, ext_fide)

    boundary.measure_posteriors(args, benign_graph_data, mask_nodes, benign_model)
    boundary.measure_posteriors(args, benign_graph_data, mask_nodes, mask_model)
    boundary.measure_posteriors(args, benign_graph_data, mask_nodes, independent_model)
    boundary.measure_posteriors(args, benign_graph_data, mask_nodes, extraction_model)


if __name__ == '__main__':
    args = parse_args()
    # ownver(args)
    verification.batch_ownver(args)
    # backdoor.run(args)
    # grove.batch_ownver(args)