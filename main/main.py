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
import backdoor
import grove
from utils.config import parse_args


def ownver(args):
    benign_graph_data, benign_model = benign.run(args)

    watermark_graph_data, watermark_nodes = boundary.mask_graph_data(args, benign_graph_data, benign_model)

    #args.mask_feat_num = 0
    _, watermark_model = benign.run(args, watermark_graph_data)

    # independent model for test
    args.benign_model = 'gcn'
    args.benign_hidden_dim = [64, 32]
    _, independent_model = benign.run(args, benign_graph_data)

    # extraction model for test
    args.extraction_model = 'gcnExtract'
    args.extraction_hidden_dim = [64, 32]
    extraction_model, _ = extraction.run(args, benign_graph_data, benign_model)


    original_posteriors = boundary.measure_posteriors(benign_graph_data, watermark_nodes, benign_model)
    independent_posteriors = boundary.measure_posteriors(benign_graph_data, watermark_nodes, independent_model)
    extraction_posteriors = boundary.measure_posteriors(benign_graph_data, watermark_nodes, extraction_model)


if __name__ == '__main__':
    args = parse_args()
    # ownver(args)
    # verification.batch_ownver(args)
    # backdoor.run(args)
    grove.batch_ownver(args)