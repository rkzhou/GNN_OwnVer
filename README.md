# Revisiting Black-box Ownership Verification for Graph Neural Networks

This paper will appear in IEEE Symposium on Security and Privacy 2024.

This repo contains code that allows you to reproduce experiments presented in the paper.

## Environment Setup

All third-party packages used in the code can be easily installed by pip or anaconda.

Package list:

pytorch, torch_geometric (need to be installed as the version which is suitable for your CUDA version, the CUDA version in our environment is 11.8)

numpy, scikit-learn, tqdm, yaml, argparse.

## File Helper

In this section, we will introduce some parameters inside files.

### In "config" Folder:

1. global_cfg.yaml:
- dataset: the graph dataset
- train_save_root: the saving path for the training models in ownership verification.
- test_save_root: the saving path for the testing models in ownership verification.
- res_path: the saving path for the ownership verification results.
2. train_setting.yaml and test_setting.yaml:
- num_model_per_arch: the number of models will be trained for each GNN model architecture. Change the number of training models in the extended studies II.

### In "main" Folder:
1. verification_cfg.py:
- process in "train_models_by_setting" function: set to "test" for the testing independent models in the extended studies I.
- classifier in "train_models_by_setting" function: set to "classifier_model" for the testing extraction models in the adaptive attacks experiments.
- attack_setting_list in "multiple_experiments" function: control which verification settings you will run.


### In "utils" Folder:

1. config.py:
- task_type: select learning paradigms, either "transductive" or "inductive".
- mask_feat_ratio: how much ratio of all nodes in the training data will be masked.
- prune_weight_ratio: how much ratio of model parameters will be pruned in robustness experiments.


## Run Experiments

### Base Experiments

Everytime after you set the corresponding parameters, you can directly run the "main.py" file under the "main" folder.

With the default settings, you will run four verification settings, with no masking magnitude, in the Cora dataset, under transductive learning.

### Robustness Experiments

Fine-tune, prune and double extraction are all implemented in the "robustness.py" file. To run all three robustness techniques, you need to select which masking magnitude of testing models you need to test for robustness. 

Besides, for the prune test, you need to set the magnitude of pruning.

After setting corresponding parameters, you can run the "robustness.py" file directly.

### Extended Studies

Set the corresponding parameters mentioned above and run the base experiment again.

### Adaptive Attacks
Set the corresponding parameters mentioned above and run the base experiment again.
