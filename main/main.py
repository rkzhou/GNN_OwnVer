import sys, os
sys.path.append(os.path.abspath('..'))
import benign
import extraction
import boundary
import main.verification as verification
from utils.config import parse_args


def ownver(args):
    benign_graph_data, benign_model = benign.run(args)

    # # independent model for test
    # args.benign_hidden_dim = [1024, 512, 256, 128]
    # _, suspicious_model = benign.run(args)
    
    watermark_graph_data, watermark_nodes = boundary.poison_graph_data(benign_graph_data, 100, 1000, benign_model, 'each_class')
    _, watermark_model = benign.run(args, watermark_graph_data)

    # extraction model for test
    suspicious_model, _ = extraction.run(args, benign_graph_data, watermark_model)

    measure_nodes = []
    for i in watermark_nodes:
        measure_nodes += i

    pair_list = list()
    for group_index in range(20):
        independent_model, extraction_model = verification.train_models(benign_graph_data, watermark_model)
        logits = verification.extract_logits(benign_graph_data, measure_nodes, independent_model, watermark_model, extraction_model)
        distance_pair = verification.measure_logits(logits)
        pair_list.append(distance_pair)
    classifier_model = verification.train_classifier(pair_list, 0.8)
    
    verification.owner_verify(benign_graph_data, watermark_model, suspicious_model, classifier_model, measure_nodes)

    # benign_posteriors = boundary.measure_posteriors(benign_graph_data, watermark_nodes, benign_model)
    # watermark_posteriors = boundary.measure_posteriors(benign_graph_data, watermark_nodes, watermark_model)
    # extraction_posteriors = boundary.measure_posteriors(benign_graph_data, watermark_nodes, suspicious_model)


if __name__ == '__main__':
    args = parse_args()
    ownver(args)