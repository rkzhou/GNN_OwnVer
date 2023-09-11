import argparse

def add_data_group(group):
    group.add_argument('--dataset', type=str, default='Cora', help="used dataset")
    group.add_argument('--data_path', type=str, default='../dataset', help="the directory used to save dataset")
    group.add_argument('--random_seed', type=int, default=0)
    group.add_argument('--verification_train_num', type=int, default=100)
    group.add_argument('--mask_node_num', type=int, default=100)
    group.add_argument('--mask_feat_num', type=int, default=1000)
    group.add_argument('--mask_type', type=str, default='each_class')


def add_benign_model_group(group):
    group.add_argument('--benign_model', type=str, default='gcn', help="used model")
    group.add_argument('--benign_train_ratio', type=float, default=0.6, help="ratio of trainset from whole dataset")
    group.add_argument('--benign_hidden_dim', nargs='+', default=[64, 32], type=int, help='hidden layers of the model')

    group.add_argument('--benign_batch_size', type=int, default=16)
    group.add_argument('--benign_train_epochs', type=int, default=1000)
    group.add_argument('--benign_lr', type=float, default=0.001)
    group.add_argument('--benign_lr_decay_steps', nargs='+', default=[500, 800], type=int)
    group.add_argument('--benign_weight_decay', type=float, default=5e-4)

    group.add_argument('--benign_train_method', type=str, default='normal')
    group.add_argument('--antidistill_train_ratio', type=float, default=0.1)


def add_atk_group(group):
    group.add_argument('--backdoor_train_node_ratio', type=float, default=0.15)
    group.add_argument('--backdoor_test_node_ratio', type=float, default=0.1)
    group.add_argument('--backdoor_feature_num', type=float, default=35)
    group.add_argument('--backdoor_target_label', type=int, default=6)

    group.add_argument('--backdoor_train_epochs', type=int, default=1000)
    group.add_argument('--backdoor_lr', type=float, default=0.001)
    group.add_argument('--backdoor_lr_decay_steps', nargs='+', default=[500, 800], type=int)
    group.add_argument('--backdoor_weight_decay', type=float, default=5e-4)


def add_extraction_model_group(group):
    group.add_argument('--extraction_model', type=str, default='sageExtract', help="used model")
    group.add_argument('--extraction_train_ratio', type=float, default=0.5, help="ratio of trainset from whole dataset")
    group.add_argument('--extraction_hidden_dim', nargs='+', default=[2048, 1024, 512], type=int, help='hidden layers of the model')

    group.add_argument('--extraction_batch_size', type=int, default=16)
    group.add_argument('--extraction_train_epochs', type=int, default=1000)
    group.add_argument('--extraction_lr', type=float, default=0.001)
    group.add_argument('--extraction_lr_decay_steps', nargs='+', default=[500, 800], type=int)
    group.add_argument('--extraction_weight_decay', type=float, default=5e-4)

    group.add_argument('--extraction_type', type=str, default='full')
    group.add_argument('--extraction_method', type=str, default='black_box')
    group.add_argument('--extraction_ratio', type=float, default=0.5)


def parse_args():
    parser = argparse.ArgumentParser()
    data_group = parser.add_argument_group(title="Data-related configuration")
    benign_model_group = parser.add_argument_group(title="Benign-model-related configuration")
    atk_group = parser.add_argument_group(title="Attack-related configuration")
    extraction_model_group = parser.add_argument_group(title="Extraction-model-related configuration")

    add_data_group(data_group)
    add_benign_model_group(benign_model_group)
    add_atk_group(atk_group)
    add_extraction_model_group(extraction_model_group)

    return parser.parse_args()