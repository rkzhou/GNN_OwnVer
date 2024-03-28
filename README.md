# Revisiting Black-box Ownership Verification for Graph Neural Networks

Paper title: Revisiting Black-box Ownership Verification for Graph Neural Networks

This paper will appear in IEEE Symposium on Security and Privacy 2024. 

This repo contains code that allows you to reproduce experiments presented in the paper.

## Environment Setup

Opearting system: Ubuntu 22.04.4 LTS

CPU: Intel i9-12900K

Graphics card: RTX 4090

RAM: 64GB

CUDA version: 11.8

You need to install some third-party libraries with the following commands:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install numpy
pip install scikit-learn
pip install tqdm
pip install pyyaml
pip install argparse
```

## File Illustration

### In "config" Folder:
1. global_cfg.yaml: 
- target_model: the architecture of target model. Valid values: [gcn, gat, sage]
- target_hidden_dims: hidden layer dimension of target model. Valid values: [[352, 128],[288, 128],[224, 128]]
- dataset: the graph dataset. Valid values: [Cora, Citeseer, Amazon, DBLP, PubMed]
- train_setting: the setting file for training local models. Valid values: fixed.
- test_setting: the setting file for training real models. Valid values: [1, 2, 3, 4]
- embedding_dim: the additional last layer in local and real models, used for reproducing Grove white-box method. Valid values: fixed.
- train_process: train local models. Valid values: fixed.
- test_process: train real models. Valid values: fixed.
- n_run: the repeating number of verification experiments. Valid values: can be any number, set for 3 in our paper.
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

```
python main.py
```

With the default settings, you will run four verification settings, with no masking magnitude, in the Cora dataset, under transductive learning.

### Robustness Experiments

Fine-tune, prune and double extraction are all implemented in the "robustness.py" file. To run all three robustness techniques, you need to select which masking magnitude of testing models you need to test for robustness. 

Besides, for the prune test, you need to set the magnitude of pruning.

After setting corresponding parameters, you can directly run the "robustness.py" file under the "main" folder.

```
python robustness.py
```

### Extended Studies

Set the corresponding parameters mentioned above and run the base experiment again.

### Adaptive Attacks

Set the corresponding parameters mentioned above and run the base experiment again.

## Results Viewing

All models and results will be saved in the path you set in the "global_cfg.yaml" file.

## Citation
